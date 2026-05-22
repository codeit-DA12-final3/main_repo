# -*- coding: utf-8 -*-
"""
Three-axis user clustering.

This script combines:
1. Sender behavior axis from Final_KimSuHyun.ipynb / sender_behavior_axis_colab.py
2. Receiver reaction axis from 수신_반응축.pdf
3. Network/environment axis from 관계망_환경축.pdf

Design choice:
- Retention/payment outcome columns are NOT used as clustering inputs.
- Count-heavy behavior and environment features are p99-winsorized, log1p transformed,
  then standardized.
- Cluster quality is checked with inertia, sampled silhouette, minimum cluster share,
  PCA projection, standardized profile, and outcome validation tables.
"""

# %% [markdown]
# ## 셀3-1. 환경설정


# %%
import ast
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def configure_utf8_output():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    for stream in [sys.stdout, sys.stderr]:
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


configure_utf8_output()


# sender_behavior_axis_colab.py와 같은 Google Drive/GitHub 경로를 사용합니다.
RUN_COLAB_SETUP = True
MOUNT_GOOGLE_DRIVE = True
SETUP_GITHUB_REPO = False

GITHUB_REPO = "codeit-DA12-final3/main_repo"
BRANCH_NAME = "feature/suhyun-analysis"
REPO_DIR = Path("/content/main_repo")

DATA_DIR = Path(
    "/content/drive/MyDrive/부트캠프/미션 및 데이터/고급 프로젝트/data"
)
OUT_DIR = Path("/content/three_axis_clustering_outputs")
WORK_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def running_in_colab():
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def mount_google_drive():
    if not running_in_colab():
        print("[LOCAL] Colab 환경이 아니므로 Google Drive mount를 건너뜁니다.")
        return False
    if not MOUNT_GOOGLE_DRIVE:
        print("MOUNT_GOOGLE_DRIVE=False 이므로 Google Drive mount를 건너뜁니다.")
        return False
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        if (Path("/content/drive/MyDrive")).exists():
            print("Google Drive mount 확인 완료: /content/drive/MyDrive")
        else:
            print("[WARN] /content/drive/MyDrive 경로가 보이지 않습니다. Drive 권한/마운트 상태를 확인하세요.")
        return True
    except Exception as e:
        print(f"[WARN] Google Drive mount skipped or failed: {e}")
        return False


if RUN_COLAB_SETUP and MOUNT_GOOGLE_DRIVE and running_in_colab():
    mount_google_drive()


if not running_in_colab():
    local_data_dir = WORK_DIR / "raw_data"
    if local_data_dir.exists():
        print(f"[LOCAL] DATA_DIR fallback: {local_data_dir}")
        DATA_DIR = local_data_dir
        OUT_DIR = WORK_DIR / "three_axis_clustering_outputs"


FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PROCESSED_DIR = OUT_DIR / "processed"

FILES = {
    "users": DATA_DIR / "accounts_user_master.csv",
    "groups": DATA_DIR / "accounts_group_raw.csv",
    "user_properties": DATA_DIR / "user_properties_raw.csv",
    "attendance_long": DATA_DIR / "accounts_attendance_long.csv",
    "payment": DATA_DIR / "accounts_paymenthistory_raw.csv",
    "question_record": DATA_DIR / "accounts_userquestionrecord_raw.csv",
    "friend_request": DATA_DIR / "accounts_friendrequest_raw.csv",
    "point_history": DATA_DIR / "accounts_pointhistory_raw.csv",
    "hackle_properties": DATA_DIR / "hackle_properties_raw.csv",
    "hackle_events": DATA_DIR / "hackle_events_raw.csv",
    "candidate_exposure": DATA_DIR / "polls_usercandidate_raw.csv",
    "question_piece": DATA_DIR / "polls_questionpiece_raw.csv",
}


# 분석 옵션
CHUNKSIZE = 1_000_000
RANDOM_STATE = 42

RUN_HACKLE_SUPPLEMENT = True
COPY_OUTPUTS_TO_REPO_IF_AVAILABLE = True
RUN_GITHUB_PUSH = False
DISPLAY_FIGURES_IN_NOTEBOOK = True
INSTALL_KOREAN_FONT_IN_COLAB = True
# 기본값은 분석, 표/그래프 저장, 화면 출력까지만 수행하고 GitHub 단계 전에 멈춥니다.
# 결과를 확인한 뒤 publish_reviewed_outputs()를 실행하면 공유 폴더 생성과 선택적 push가 진행됩니다.
STOP_BEFORE_GITHUB_STEP_FOR_REVIEW = True

MAX_THREE_AXIS_CLUSTER_K = 8
# 기본값은 elbow/drop 지표와 실루엣 점수를 함께 보고 k를 자동 선정합니다.
# 그래프 확인 후 수동으로 고정하려면 원하는 숫자를 넣으세요. 예: SELECTED_THREE_AXIS_K = 3
SELECTED_THREE_AXIS_K = None
COMPARE_THREE_AXIS_KS = [4]
REQUIRE_MANUAL_K_SELECTION = False
PREFERRED_AUTO_K_RANGE = (3, 6)
AUTO_K_ELBOW_WEIGHT = 0.45
AUTO_K_SILHOUETTE_WEIGHT = 0.35
AUTO_K_DROP_WEIGHT = 0.20
MIN_CLUSTER_SHARE = 0.01
SILHOUETTE_SAMPLE_SIZE = 10_000
PCA_PLOT_SAMPLE_SIZE = 50_000
USE_MINIBATCH_THRESHOLD = 200_000

HACKLE_QUESTION_START_KEYS = [
    "click_question_open",
    "view_questions_tap",
    "click_bottom_navigation_questions",
]
HACKLE_QUESTION_COMPLETE_KEYS = [
    "complete_question",
    "click_question_complete",
    "submit_question",
    "complete_vote",
    "click_vote_complete",
]
HACKLE_QUESTION_SKIP_KEYS = ["skip_question"]

WINDOWS = {
    "d0": (0, 0),
    "d0_3": (0, 3),
    "d0_7": (0, 7),
    "d0_28": (0, 28),
    "d1_28": (1, 28),
    "d8_28": (8, 28),
}

SENDER_FEATURES = [
    "sent_vote_count_d0_7",
    "sent_vote_active_days_d0_7",
    "sent_vote_distinct_chosen_users_d0_7",
    "sent_vote_distinct_questions_d0_7",
    "sent_vote_complete_rate_d0_7",
    "hackle_question_start_count_d0_7",
    "hackle_question_completion_rate_d0_7",
]

RECEIVER_FEATURES = [
    "received_vote_count_d0_7",
    "received_vote_days_d0_7",
    "received_vote_count_d0_3",
    "received_vote_read_count_d0_7",
    "received_vote_read_rate_d0_7",
    "unread_received_vote_count_d0_7",
    "early_received_score_d0_7",
]

NETWORK_FEATURES = [
    "friend_count",
    "same_class_active_d0_7",
    "same_grade_active_d0_7",
    "candidate_exposure_count_d0_7",
    "candidate_unique_question_count_d0_7",
    "candidate_to_recv_rate_d0_7",
    "pending_votes",
]

PROFILE_ONLY_FEATURES = [
    "same_school_joined_user_count",
    "same_class_total_users",
    "same_grade_total_users",
    "same_class_active_ratio_d0_7",
    "same_grade_active_ratio_d0_7",
    "candidate_exposure_days_d0_7",
]

OUTCOME_COLUMNS = [
    "retention_d1_28",
    "retention_d8_28",
    "paid_flag_d0_28",
    "paid_flag_d8_28",
    "payment_count_d0_28",
    "payment_count_d8_28",
    "payment_amount_d0_28",
    "payment_amount_d8_28",
    "active_days_d0_28",
    "active_days_d8_28",
    "attendance_after_received_vote_yn_d0_28",
    "point_use_after_received_vote_yn_d0_28",
    "payment_after_received_vote_yn_d0_28",
]

LABEL_MAP = {
    "sent_vote_count_d0_7": "D0-D7 보낸 투표 수",
    "sent_vote_active_days_d0_7": "D0-D7 발신 활동일수",
    "sent_vote_distinct_chosen_users_d0_7": "D0-D7 선택한 유저 수",
    "sent_vote_distinct_questions_d0_7": "D0-D7 발신 질문 수",
    "sent_vote_complete_rate_d0_7": "D0-D7 발신 완료율",
    "hackle_question_start_count_d0_7": "D0-D7 Hackle 질문 시작",
    "hackle_question_completion_rate_d0_7": "D0-D7 Hackle 완료율",
    "received_vote_count_d0_7": "D0-D7 받은 투표 수",
    "received_vote_days_d0_7": "D0-D7 수신 발생일수",
    "received_vote_count_d0_3": "D0-D3 받은 투표 수",
    "received_vote_read_count_d0_7": "D0-D7 받은 투표 열람 수",
    "received_vote_read_rate_d0_7": "D0-D7 수신 열람률",
    "unread_received_vote_count_d0_7": "D0-D7 미열람 수신 수",
    "early_received_score_d0_7": "초기 수신 빠름 점수",
    "friend_count": "친구 수",
    "same_class_active_d0_7": "같은 반 활성 유저 수",
    "same_grade_active_d0_7": "같은 학년 활성 유저 수",
    "candidate_exposure_count_d0_7": "D0-D7 후보 노출 수",
    "candidate_unique_question_count_d0_7": "D0-D7 후보 노출 질문 수",
    "candidate_to_recv_rate_d0_7": "후보 노출 대비 선택률",
    "pending_votes": "미확인 Ping 수",
    "retention_d1_28": "D1-D28 재방문율",
    "retention_d8_28": "D8-D28 재방문율",
    "paid_flag_d0_28": "D0-D28 결제율",
    "paid_flag_d8_28": "D8-D28 결제율",
    "active_days_d0_28": "D0-D28 활동일수",
    "active_days_d8_28": "D8-D28 활동일수",
}

ANALYSIS_NOTES = []
FIGURE_CAPTION_ROWS = []
GRAPH_VALIDATION_ROWS = []
FINAL_INTERPRETATION_MARKDOWN = ""
FINAL_INTERPRETATION_ROWS = []
KOREAN_FONT_NAME = None
LAST_CLUSTER_BASE = None
LAST_OUTCOME = None
LAST_K_METRICS = None

FIGURE_CAPTIONS = {
    "three_axis_cluster_k_selection": (
        "k 후보별 군집 내 제곱합, elbow score, 표본 실루엣 점수, 자동 k 선택 점수, 최소 군집 비율을 함께 본다. "
        "선택된 k는 빨간 점선이며, elbow가 뚜렷하면서 실루엣 점수도 높은 후보인지 확인한다."
    ),
    "three_axis_cluster_pca_2d": (
        "표준화된 3축 입력 변수를 PCA 2차원에 투영한 그림이다. "
        "점들이 완전히 분리되지 않아도, 색상별 군집이 어느 방향으로 밀집하는지로 세그먼트 구분 가능성을 확인한다."
    ),
    "three_axis_cluster_profile_heatmap": (
        "군집별 표준화 입력 변수 평균이다. 0보다 크면 전체 평균보다 높은 축/행동이고, "
        "0보다 작으면 전체 평균보다 낮은 축/행동이다. 군집명 해석의 핵심 근거로 사용한다."
    ),
    "three_axis_cluster_outcome_rates": (
        "재방문율, 결제율, 수신 이후 출석률은 클러스터링 입력에 넣지 않고 사후 검증으로만 비교한다. "
        "따라서 이 그래프는 군집이 실제 가치 지표와 연결되는지 확인하는 용도다."
    ),
    "three_axis_cluster_payment_count": (
        "군집별 결제 빈도 평균을 전체 유저 기준과 결제자 기준으로 분리해 본다. "
        "결제율은 낮아도 결제자 평균 빈도가 높으면 소수 고가치 유저가 몰린 군집일 수 있다."
    ),
    "three_axis_cluster_payment_amount": (
        "군집별 추정 구매 하트 수 평균을 전체 유저 기준과 결제자 기준으로 비교한다. "
        "결제율과 구매 규모를 함께 봐야 수익화 기여도를 과소/과대 해석하지 않는다."
    ),
    "three_axis_cluster_active_days": (
        "군집별 누적 활동일수를 전체 유저 평균과 활동 유저 평균으로 나누어 본다. "
        "전체 평균은 활동 진입률의 영향을, 활동 유저 평균은 진입 후 활동 강도의 영향을 함께 반영한다."
    ),
    "three_axis_cluster_axis_scores": (
        "군집별 발신, 수신, 관계망 환경 축 점수 평균을 비교한다. "
        "어떤 축이 군집명을 결정하는 핵심 특징인지 확인하는 그래프다."
    ),
    "three_axis_cluster_outcome_heatmap": (
        "군집별 핵심 결과 지표를 한 화면에서 비교한다. "
        "재방문, 결제, 활동일수 중 어느 성과가 특정 군집에서 두드러지는지 확인한다."
    ),
    "three_axis_segment_cluster_heatmap": (
        "각 강도/친구 수 구간 안에서 어떤 클러스터가 많이 나타나는지 보는 구성비 히트맵이다. "
        "특정 구간이 특정 클러스터에 몰리면 해당 축의 강도가 군집 해석에 크게 기여한 것으로 볼 수 있다."
    ),
}


# %% [markdown]
# ## 셀3-2. 공통 함수


# %%
def setup_colab_environment():
    if not (RUN_COLAB_SETUP and running_in_colab()):
        return
    mount_google_drive()

    if SETUP_GITHUB_REPO:
        try:
            from google.colab import userdata

            token = userdata.get("GITHUB_TOKEN")
            if not token:
                print("[WARN] GITHUB_TOKEN is missing. GitHub clone skipped.")
                return
            Path("/root/.netrc").write_text(
                f"machine github.com\nlogin x-access-token\npassword {token}\n",
                encoding="utf-8",
            )
            os.chmod("/root/.netrc", 0o600)
            if REPO_DIR.exists() and not (REPO_DIR / ".git").exists():
                shutil.rmtree(REPO_DIR)
            if not REPO_DIR.exists():
                subprocess.run(
                    ["git", "clone", f"https://github.com/{GITHUB_REPO}.git", str(REPO_DIR)],
                    check=True,
                )
            subprocess.run(["git", "switch", BRANCH_NAME], cwd=REPO_DIR, check=False)
        except Exception as e:
            print(f"[WARN] GitHub repo setup skipped or failed: {e}")


def init_output_dirs():
    for path in [OUT_DIR, FIG_DIR, TABLE_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def install_colab_korean_font_if_needed():
    if not (running_in_colab() and INSTALL_KOREAN_FONT_IN_COLAB):
        return
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if {"NanumGothic", "NanumBarunGothic"} & available_fonts:
        return
    try:
        print("Colab 한글 폰트를 설치합니다: fonts-nanum")
        subprocess.run(["apt-get", "-qq", "update"], check=False)
        subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-nanum"], check=True)
        subprocess.run(["fc-cache", "-fv"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        matplotlib_cache_dir = Path.home() / ".cache" / "matplotlib"
        shutil.rmtree(matplotlib_cache_dir, ignore_errors=True)
        font_manager.fontManager = font_manager._load_fontmanager(try_read_cache=False)
        for font_path in [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        ]:
            if Path(font_path).exists():
                font_manager.fontManager.addfont(font_path)
        print("Colab 한글 폰트 설치/등록 완료")
    except Exception as exc:
        print(f"[WARN] Colab 한글 폰트 설치를 건너뜁니다: {exc}")


def register_korean_font_from_file():
    candidate_paths = [
        Path("C:/Windows/Fonts/NotoSansCJKkr-Regular.otf"),
        Path("C:/Windows/Fonts/NotoSansKR-Regular.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf"),
    ]
    for font_path in candidate_paths:
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            return font_manager.FontProperties(fname=str(font_path)).get_name()
        except Exception:
            continue
    return None


def find_available_korean_font_name():
    registered_font = register_korean_font_from_file()
    if registered_font:
        return registered_font

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
        "NanumGothic",
        "NanumBarunGothic",
        "AppleGothic",
    ]
    for font_name in preferred_fonts:
        if font_name not in available_fonts:
            continue
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=[font_name]),
                fallback_to_default=False,
            )
            return font_name
        except Exception:
            continue
    return None


def setup_plot_style():
    global KOREAN_FONT_NAME
    install_colab_korean_font_if_needed()
    selected_font = find_available_korean_font_name()
    sns_rc = {"axes.unicode_minus": False}
    if selected_font:
        KOREAN_FONT_NAME = selected_font
        sns_rc.update(
            {
                "font.family": [selected_font],
                "font.sans-serif": [selected_font],
            }
        )
        sns.set_theme(style="whitegrid", rc=sns_rc)
        plt.rcParams.update(sns_rc)
        matplotlib.rcParams.update(sns_rc)
        print(f"그래프 한글 폰트 설정: {selected_font}")
    else:
        sns.set_theme(style="whitegrid", rc=sns_rc)
        plt.rcParams.update(sns_rc)
        print("[WARN] 한글 지원 폰트를 찾지 못했습니다. 그래프 한글이 네모/깨짐으로 표시될 수 있습니다.")
        print("[WARN] Colab에서는 런타임을 다시 시작한 뒤 셀3-1, 셀3-2, 셀3-11 순서로 다시 실행해 주세요.")


def ensure_korean_font():
    global KOREAN_FONT_NAME
    if not KOREAN_FONT_NAME:
        KOREAN_FONT_NAME = find_available_korean_font_name()
    if KOREAN_FONT_NAME:
        font_rc = {
            "font.family": [KOREAN_FONT_NAME],
            "font.sans-serif": [KOREAN_FONT_NAME],
            "axes.unicode_minus": False,
        }
        plt.rcParams.update(font_rc)
        matplotlib.rcParams.update(font_rc)
    return KOREAN_FONT_NAME


def apply_korean_font_to_figure(fig):
    font_name = ensure_korean_font()
    if not font_name:
        return
    for text_obj in fig.findobj(match=lambda obj: hasattr(obj, "set_fontfamily")):
        try:
            text_obj.set_fontfamily(font_name)
        except Exception:
            continue


def read_csv(path, **kwargs):
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def as_user_id(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def normalize_id(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def to_day(series):
    return pd.to_datetime(series, errors="coerce").dt.floor("D")


def parse_list_len(value):
    if pd.isna(value):
        return 0
    if isinstance(value, list):
        return len(value)
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def window_mask(day_offset, name):
    start, end = WINDOWS[name]
    return day_offset.between(start, end, inclusive="both")


def add_signup_offset(df, user_col, date_col, users):
    lookup = users[["user_id", "signup_date"]].drop_duplicates()
    out = df.merge(lookup, left_on=user_col, right_on="user_id", how="left", suffixes=("", "_signup"))
    if user_col != "user_id":
        out = out.drop(columns=["user_id"]).rename(columns={user_col: "user_id"})
    out["day_offset"] = (out[date_col] - out["signup_date"]).dt.days
    return out


def compact_numeric_fill(df):
    out = df.copy()
    for col in out.columns:
        if col == "user_id":
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
    return out


def fill_bool_na_false(series):
    return series.fillna(False).astype(bool)


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0)
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def rate(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(bool).mean() * 100


def nonzero_count(series):
    return int((pd.to_numeric(series, errors="coerce").fillna(0) > 0).sum())


def add_paid_only_payment_means(summary):
    out = summary.copy()
    for window in ["d0_28", "d8_28"]:
        paid_user_count_col = f"payment_{window}_paid_user_count"
        if paid_user_count_col not in out.columns:
            continue
        denom = pd.to_numeric(out[paid_user_count_col], errors="coerce").replace(0, np.nan)

        count_sum_col = f"payment_{window}_count_sum"
        count_paid_mean_col = f"payment_{window}_count_paid_user_mean"
        if count_sum_col in out.columns:
            count_sum = pd.to_numeric(out[count_sum_col], errors="coerce").fillna(0)
            out[count_paid_mean_col] = (count_sum / denom).fillna(0)

        amount_sum_col = f"payment_{window}_amount_sum"
        amount_paid_mean_col = f"payment_{window}_amount_paid_user_mean"
        if amount_sum_col in out.columns:
            amount_sum = pd.to_numeric(out[amount_sum_col], errors="coerce").fillna(0)
            out[amount_paid_mean_col] = (amount_sum / denom).fillna(0)
    return out


def add_nonzero_user_means(summary):
    out = summary.copy()
    denominator_pairs = [
        ("active_days_d0_28_sum", "active_days_d0_28_active_user_count", "active_days_d0_28_active_user_mean"),
        ("active_days_d8_28_sum", "active_days_d8_28_active_user_count", "active_days_d8_28_active_user_mean"),
        ("sent_vote_count_d0_7_sum", "sent_vote_user_count_d0_7", "sent_vote_count_d0_7_sender_mean"),
        (
            "received_vote_count_d0_7_sum",
            "received_vote_user_count_d0_7",
            "received_vote_count_d0_7_receiver_mean",
        ),
        (
            "candidate_exposure_count_d0_7_sum",
            "candidate_exposed_user_count_d0_7",
            "candidate_exposure_count_d0_7_exposed_user_mean",
        ),
    ]
    for numerator_col, denominator_col, output_col in denominator_pairs:
        if {numerator_col, denominator_col}.issubset(out.columns):
            numerator = pd.to_numeric(out[numerator_col], errors="coerce").fillna(0)
            denominator = pd.to_numeric(out[denominator_col], errors="coerce").replace(0, np.nan)
            out[output_col] = (numerator / denominator).fillna(0)
    return out


def add_adjusted_mean_columns(summary):
    return add_nonzero_user_means(add_paid_only_payment_means(summary))


def label_for(col):
    return LABEL_MAP.get(col, col)


def cell_note(title, purpose, output):
    ANALYSIS_NOTES.append({"title": title, "purpose": purpose, "output": output})
    print(f"\n[{title}]")
    print(f"- 목적: {purpose}")
    print(f"- 산출물: {output}")


def save_table(df, name):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved table: {path}")
    return path


def save_processed(df, name):
    path = PROCESSED_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved processed: {path}")
    return path


def display_png_if_possible(path):
    if not DISPLAY_FIGURES_IN_NOTEBOOK:
        return
    try:
        from IPython import get_ipython
        from IPython.display import Image, display

        if get_ipython() is None:
            return
        print(f"[그래프 출력] {path.name}")
        display(Image(filename=str(path)))
    except Exception as exc:
        print(f"[WARN] 그래프 화면 출력 생략: {exc}")


def display_all_saved_figures():
    if not FIG_DIR.exists():
        print(f"그래프 폴더가 아직 없습니다: {FIG_DIR}")
        return
    figure_paths = sorted(FIG_DIR.glob("*.png"))
    if not figure_paths:
        print(f"저장된 PNG 그래프가 없습니다: {FIG_DIR}")
        return
    for path in figure_paths:
        display_png_if_possible(path)


def display_dataframe_if_possible(df, title, max_rows=20):
    print(f"\n[{title}]")
    if df is None or len(df) == 0:
        print("표시할 데이터가 없습니다.")
        return
    preview = df.head(max_rows).copy()
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(preview)
            return
    except Exception:
        pass
    print(preview.to_string(index=False))


def display_analysis_results(outcome, k_metrics):
    selected_cols = [
        "k",
        "selected_k",
        "auto_suggested_k",
        "auto_k_score",
        "elbow_score",
        "inertia_drop_pct_from_prev",
        "silhouette_score",
        "min_cluster_share_pct",
        "selection_reason",
    ]
    display_dataframe_if_possible(
        k_metrics[[c for c in selected_cols if c in k_metrics.columns]],
        "k 선택 결과",
        max_rows=len(k_metrics),
    )
    display_dataframe_if_possible(outcome, "클러스터별 결과 요약", max_rows=30)
    if FINAL_INTERPRETATION_ROWS:
        interpretation_cols = [
            "cluster_label",
            "user_count",
            "cluster_share_pct",
            "dominant_axis",
            "retention_d1_28_rate",
            "payment_d0_28_rate",
            "active_days_d0_28_mean",
            "interpretation",
        ]
        interpretation_df = pd.DataFrame(FINAL_INTERPRETATION_ROWS)
        display_dataframe_if_possible(
            interpretation_df[[c for c in interpretation_cols if c in interpretation_df.columns]],
            "클러스터 자동 해석",
            max_rows=30,
        )
    print("\n[결과 저장 위치]")
    print(f"- figures: {FIG_DIR}")
    print(f"- tables: {TABLE_DIR}")
    print(f"- processed: {PROCESSED_DIR}")


def save_fig(fig, name, interpretation=None):
    path = FIG_DIR / f"{name}.png"
    apply_korean_font_to_figure(fig)
    interpretation = interpretation or FIGURE_CAPTIONS.get(name, "")
    if interpretation:
        wrapped = "\n".join(textwrap.wrap(str(interpretation), width=95, break_long_words=False))
        line_count = max(1, wrapped.count("\n") + 1)
        bottom = min(0.32, 0.08 + line_count * 0.035)
        fig.text(
            0.01,
            0.015,
            f"그래프 해석: {wrapped}",
            ha="left",
            va="bottom",
            fontsize=9,
            color="#333333",
            bbox={"facecolor": "#F7F7F7", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.45"},
        )
        fig.tight_layout(rect=(0, bottom, 1, 1))
        FIGURE_CAPTION_ROWS.append(
            {
                "figure_file": path.name,
                "figure_key": name,
                "interpretation": interpretation,
            }
        )
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {path}")
    display_png_if_possible(path)
    return path


def save_observation_window_definition():
    rows = [
        {
            "metric_group": "retention",
            "metric": "retention_d1_28",
            "window": "D1-D28",
            "definition": "가입 다음날부터 28일 이내 출석 기록이 1회 이상 있는지",
            "used_in_clustering": False,
            "role": "군집 사후 검증 결과 지표",
        },
        {
            "metric_group": "retention",
            "metric": "retention_d8_28",
            "window": "D8-D28",
            "definition": "가입 8일부터 28일 이내 출석 기록이 1회 이상 있는지",
            "used_in_clustering": False,
            "role": "군집 사후 검증 결과 지표",
        },
        {
            "metric_group": "payment",
            "metric": "paid_flag/payment_count/payment_amount_d0_28",
            "window": "D0-D28",
            "definition": "가입일부터 28일 이내 결제 여부, 결제 횟수, 추정 구매 하트 수",
            "used_in_clustering": False,
            "role": "군집 사후 검증 결과 지표",
        },
        {
            "metric_group": "payment",
            "metric": "paid_flag/payment_count/payment_amount_d8_28",
            "window": "D8-D28",
            "definition": "가입 8일부터 28일 이내 결제 여부, 결제 횟수, 추정 구매 하트 수",
            "used_in_clustering": False,
            "role": "군집 사후 검증 결과 지표",
        },
        {
            "metric_group": "active_days",
            "metric": "active_days_d0_28",
            "window": "D0-D28",
            "definition": "가입일부터 28일 이내 발신, 수신 열람, 친구 요청/수락, 포인트 사용이 발생한 고유 날짜 수",
            "used_in_clustering": False,
            "role": "군집 사후 검증 결과 지표",
        },
        {
            "metric_group": "active_days",
            "metric": "active_days_d8_28",
            "window": "D8-D28",
            "definition": "가입 8일부터 28일 이내 발신, 수신 열람, 친구 요청/수락, 포인트 사용이 발생한 고유 날짜 수",
            "used_in_clustering": False,
            "role": "군집 사후 검증 결과 지표",
        },
        {
            "metric_group": "clustering_input",
            "metric": "sender/receiver/network features",
            "window": "D0-D7",
            "definition": "초기 행동/환경 관찰 기간. 발신, 수신, 후보 노출, 같은 반/학년 활성도는 이 기간으로 집계",
            "used_in_clustering": True,
            "role": "클러스터링 입력 변수",
        },
    ]
    return save_table(pd.DataFrame(rows), "three_axis_observation_window_definition")


def check_source_references():
    required_columns = {
        "users": ["id", "created_at", "friend_id_list", "group_id", "pending_votes"],
        "groups": ["id", "grade", "class_num", "school_id"],
        "user_properties": ["user_id", "class", "grade", "school_id"],
        "attendance_long": ["user_id", "attendance_date_list"],
        "payment": ["id", "productId", "created_at", "user_id"],
        "question_record": [
            "id",
            "status",
            "created_at",
            "chosen_user_id",
            "question_id",
            "user_id",
            "has_read",
            "answer_updated_at",
            "opened_times",
        ],
        "friend_request": ["id", "status", "created_at", "updated_at", "receive_user_id", "send_user_id"],
        "point_history": ["id", "delta_point", "created_at", "user_id"],
        "hackle_properties": ["session_id", "user_id"],
        "hackle_events": ["event_id", "event_datetime", "event_key", "session_id"],
        "candidate_exposure": ["id", "created_at", "question_piece_id", "user_id"],
        "question_piece": ["id", "question_id"],
    }
    rows = []
    for file_key, cols in required_columns.items():
        path = FILES[file_key]
        exists = path.exists()
        actual_cols = []
        error = ""
        if exists:
            try:
                actual_cols = read_csv(path, nrows=0).columns.tolist()
            except Exception as exc:
                error = repr(exc)
        missing = [col for col in cols if col not in actual_cols]
        rows.append(
            {
                "file_key": file_key,
                "path": str(path),
                "exists": exists,
                "required_columns": ", ".join(cols),
                "missing_columns": ", ".join(missing),
                "actual_columns": ", ".join(actual_cols),
                "status": "ok" if exists and not missing and not error else "error",
                "error": error,
            }
        )
    result = pd.DataFrame(rows)
    save_table(result, "three_axis_source_reference_check")
    failed = result.loc[result["status"].ne("ok")]
    if not failed.empty:
        raise ValueError(
            "원천 파일/컬럼 참조 검증 실패: "
            + "; ".join(f"{row.file_key} missing [{row.missing_columns}]" for row in failed.itertuples())
        )
    return result


def validate_graph_data(figure_key, df, required_cols=None, value_cols=None, min_rows=1, allow_all_zero=True):
    required_cols = required_cols or []
    value_cols = value_cols or []
    missing_cols = [col for col in required_cols + value_cols if col not in df.columns]
    row_count = len(df)
    non_null_summary = {}
    numeric_abs_sum = 0.0
    for col in value_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            non_null_summary[col] = int(values.notna().sum())
            numeric_abs_sum += float(values.fillna(0).abs().sum())
    status = "ok"
    message = ""
    if missing_cols:
        status = "error"
        message = f"누락 컬럼: {missing_cols}"
    elif row_count < min_rows:
        status = "error"
        message = f"행 수 {row_count} < 최소 행 수 {min_rows}"
    elif value_cols and numeric_abs_sum == 0:
        status = "warn" if allow_all_zero else "error"
        message = "그래프에 표시할 숫자 값이 모두 0입니다."

    GRAPH_VALIDATION_ROWS.append(
        {
            "figure_key": figure_key,
            "status": status,
            "row_count": row_count,
            "required_columns": ", ".join(required_cols),
            "value_columns": ", ".join(value_cols),
            "missing_columns": ", ".join(missing_cols),
            "numeric_abs_sum": numeric_abs_sum,
            "non_null_value_counts": str(non_null_summary),
            "message": message,
        }
    )
    if status == "error":
        raise ValueError(f"{figure_key} 그래프 데이터 검증 실패: {message}")
    return status


def annotate_all_zero_if_needed(ax, status):
    if status == "warn":
        ax.text(
            0.5,
            0.5,
            "표시할 값이 모두 0입니다.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#555555",
            bbox={"facecolor": "#FFFFFF", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.5"},
        )


def winsorize_series(series, q=0.99):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    upper = s.quantile(q)
    if pd.isna(upper):
        return s
    return s.clip(lower=0, upper=upper)


def concat_grouped_sum(parts, all_users):
    if not parts:
        return pd.DataFrame({"user_id": all_users["user_id"].unique()})
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return pd.DataFrame({"user_id": all_users["user_id"].unique()})
    return out.groupby("user_id", as_index=False).sum(numeric_only=True)


def build_model(k, n_rows):
    if n_rows >= USE_MINIBATCH_THRESHOLD:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
            batch_size=8192,
            reassignment_ratio=0.01,
        )
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)


def sampled_indices(n_rows, sample_size, random_state):
    if n_rows <= sample_size:
        return np.arange(n_rows)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


def unique_labels(label_map):
    seen = {}
    out = {}
    for key, label in label_map.items():
        label = str(label).strip() or f"군집 {key}"
        seen[label] = seen.get(label, 0) + 1
        out[key] = label if seen[label] == 1 else f"{label} {seen[label]}"
    return out


# %% [markdown]
# ## 셀3-3. 원천 데이터별 지표 생성 함수


# %%
def build_users_population():
    cell_note(
        "셀 01. 유저 모집단 정의",
        "가입일, 친구 수, 학교/학년/반 정보를 유저 단위로 정리하고 홈화면 진입 조건 모집단을 고정합니다.",
        "population_funnel.csv, 01_users_population.csv",
    )
    users = read_csv(
        FILES["users"],
        usecols=["id", "created_at", "friend_id_list", "group_id", "pending_votes"],
        dtype={"id": "string", "group_id": "string"},
    )
    users = users.drop_duplicates(subset=["id"]).rename(columns={"id": "user_id"})
    users["user_id"] = as_user_id(users["user_id"])
    users["signup_at"] = pd.to_datetime(users["created_at"], errors="coerce")
    users["signup_date"] = users["signup_at"].dt.floor("D")
    users["friend_count"] = users["friend_id_list"].apply(parse_list_len)
    users["pending_votes"] = pd.to_numeric(users["pending_votes"], errors="coerce").fillna(0)
    users["group_id"] = normalize_id(users["group_id"])
    users = users.dropna(subset=["user_id", "signup_date"])

    groups = read_csv(
        FILES["groups"],
        usecols=["id", "grade", "class_num", "school_id"],
        dtype={"id": "string", "grade": "string", "class_num": "string", "school_id": "string"},
    )
    groups = groups.rename(
        columns={
            "id": "group_id",
            "grade": "grade_from_group",
            "class_num": "class_from_group",
            "school_id": "school_id_from_group",
        }
    )
    groups["group_id"] = normalize_id(groups["group_id"])
    groups["school_id_from_group"] = normalize_id(groups["school_id_from_group"])
    groups = groups.drop_duplicates(subset=["group_id"])

    props = read_csv(
        FILES["user_properties"],
        usecols=["user_id", "class", "grade", "school_id"],
        dtype={"user_id": "string", "class": "string", "grade": "string", "school_id": "string"},
    )
    props["user_id"] = as_user_id(props["user_id"])
    props = props.rename(
        columns={
            "class": "class_from_properties",
            "grade": "grade_from_properties",
            "school_id": "school_id_from_properties",
        }
    )
    props["school_id_from_properties"] = normalize_id(props["school_id_from_properties"])
    props = props.drop_duplicates(subset=["user_id"])

    users = users.merge(groups, on="group_id", how="left").merge(props, on="user_id", how="left")
    users["school_id"] = users["school_id_from_group"].fillna(users["school_id_from_properties"])
    users["grade"] = users["grade_from_group"].fillna(users["grade_from_properties"])
    users["class_num"] = users["class_from_group"].fillna(users["class_from_properties"])

    school_user_counts = (
        users.dropna(subset=["school_id"])
        .groupby("school_id", as_index=False)
        .agg(same_school_joined_user_count=("user_id", "nunique"))
    )
    users = users.merge(school_user_counts, on="school_id", how="left")
    users["same_school_joined_user_count"] = users["same_school_joined_user_count"].fillna(0).astype(int)

    users["grade_key"] = (
        users["school_id"].astype("string").fillna("")
        + "|"
        + users["grade"].astype("string").fillna("")
    )
    users["class_key"] = users["grade_key"] + "|" + users["class_num"].astype("string").fillna("")
    users.loc[users["school_id"].isna() | users["grade"].isna(), "grade_key"] = pd.NA
    users.loc[users["school_id"].isna() | users["grade"].isna() | users["class_num"].isna(), "class_key"] = pd.NA

    class_counts = (
        users.dropna(subset=["class_key"])
        .groupby("class_key", as_index=False)
        .agg(same_class_total_users=("user_id", "nunique"))
    )
    grade_counts = (
        users.dropna(subset=["grade_key"])
        .groupby("grade_key", as_index=False)
        .agg(same_grade_total_users=("user_id", "nunique"))
    )
    users = users.merge(class_counts, on="class_key", how="left").merge(grade_counts, on="grade_key", how="left")
    users["same_class_total_users"] = users["same_class_total_users"].fillna(0)
    users["same_grade_total_users"] = users["same_grade_total_users"].fillna(0)

    users["home_entry_population"] = (users["friend_count"] >= 4) & (users["same_school_joined_user_count"] >= 40)

    population_funnel = pd.DataFrame(
        {
            "population": ["all_users", "has_signup_date", "home_entry_population"],
            "user_count": [
                users["user_id"].nunique(),
                users.dropna(subset=["signup_date"])["user_id"].nunique(),
                users.loc[users["home_entry_population"], "user_id"].nunique(),
            ],
        }
    )
    save_table(population_funnel, "population_funnel")

    keep_cols = [
        "user_id",
        "signup_at",
        "signup_date",
        "friend_count",
        "pending_votes",
        "group_id",
        "school_id",
        "grade",
        "class_num",
        "grade_key",
        "class_key",
        "same_school_joined_user_count",
        "same_class_total_users",
        "same_grade_total_users",
        "home_entry_population",
    ]
    save_processed(users[keep_cols], "01_users_population.csv")
    return users[keep_cols].copy()


def build_attendance_metrics(users):
    cell_note(
        "셀 02. 출석 기반 재방문 지표",
        "가입일 기준 D1-D28, D8-D28 기간 출석 여부를 재방문 결과 지표로 만듭니다.",
        "02_attendance_metrics.csv",
    )
    attendance = read_csv(
        FILES["attendance_long"],
        usecols=["user_id", "attendance_date_list"],
        dtype={"user_id": "string"},
        parse_dates=["attendance_date_list"],
    )
    attendance = attendance.rename(columns={"attendance_date_list": "attendance_date"})
    attendance["user_id"] = as_user_id(attendance["user_id"])
    attendance["attendance_date"] = pd.to_datetime(attendance["attendance_date"], errors="coerce").dt.floor("D")
    attendance = attendance.dropna(subset=["user_id", "attendance_date"]).drop_duplicates(["user_id", "attendance_date"])
    attendance = add_signup_offset(attendance, "user_id", "attendance_date", users)

    metrics = pd.DataFrame({"user_id": users["user_id"].unique()})
    for name in ["d1_28", "d8_28"]:
        part = (
            attendance.loc[window_mask(attendance["day_offset"], name)]
            .groupby("user_id", as_index=False)
            .agg(**{f"attendance_days_{name}": ("attendance_date", "nunique")})
        )
        metrics = metrics.merge(part, on="user_id", how="left")
    metrics = compact_numeric_fill(metrics)
    metrics["retention_d1_28"] = metrics["attendance_days_d1_28"] >= 1
    metrics["retention_d8_28"] = metrics["attendance_days_d8_28"] >= 1
    save_processed(metrics, "02_attendance_metrics.csv")
    return metrics, attendance[["user_id", "attendance_date", "day_offset"]].copy()


def build_payment_metrics(users):
    cell_note(
        "셀 03. 결제 결과 지표",
        "가입일 기준 D0-D28, D8-D28 결제 빈도/여부와 결제 규모 후보를 계산합니다.",
        "payment_amount_source_check.csv, 03_payment_metrics.csv",
    )
    header = read_csv(FILES["payment"], nrows=0).columns.tolist()
    base_cols = [c for c in ["id", "productId", "created_at", "user_id"] if c in header]
    amount_candidates = [
        "amount",
        "price",
        "paid_amount",
        "payment_amount",
        "total_amount",
        "total_price",
        "purchase_amount",
        "revenue",
        "sales_amount",
        "gross_amount",
        "net_amount",
        "money",
        "cash",
        "value",
        "paidAmount",
        "paymentAmount",
        "totalAmount",
        "totalPrice",
    ]
    header_lookup = {str(col).lower(): col for col in header}
    amount_col = next((header_lookup[c.lower()] for c in amount_candidates if c.lower() in header_lookup), None)
    usecols = base_cols + ([amount_col] if amount_col else [])

    payment = read_csv(
        FILES["payment"],
        usecols=usecols,
        dtype={"id": "string", "productId": "string", "user_id": "string"},
    )
    if "id" in payment.columns:
        payment = payment.drop_duplicates(subset=["id"])
    payment["user_id"] = as_user_id(payment["user_id"])
    payment["payment_date"] = to_day(payment["created_at"])
    payment = payment.dropna(subset=["user_id", "payment_date"])
    payment = add_signup_offset(payment, "user_id", "payment_date", users)

    payment["product_heart_qty"] = pd.to_numeric(
        payment.get("productId", pd.Series("", index=payment.index)).astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).fillna(0)
    if amount_col:
        amount_from_col = pd.to_numeric(
            payment[amount_col].astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0)
        payment["payment_amount_value"] = amount_from_col
        payment_amount_source = amount_col
    else:
        payment["payment_amount_value"] = payment["product_heart_qty"]
        payment_amount_source = "productId_digit_as_estimated_heart_qty"

    metrics = pd.DataFrame({"user_id": users["user_id"].unique()})
    for name in ["d0_28", "d8_28"]:
        part = (
            payment.loc[window_mask(payment["day_offset"], name)]
            .groupby("user_id", as_index=False)
            .agg(
                **{
                    f"payment_count_{name}": ("payment_date", "size"),
                    f"payment_active_days_{name}": ("payment_date", "nunique"),
                    f"payment_amount_{name}": ("payment_amount_value", "sum"),
                }
            )
        )
        metrics = metrics.merge(part, on="user_id", how="left")
    metrics = compact_numeric_fill(metrics)
    metrics["paid_flag_d0_28"] = metrics["payment_count_d0_28"] > 0
    metrics["paid_flag_d8_28"] = metrics["payment_count_d8_28"] > 0

    source_check = pd.DataFrame(
        [{"payment_amount_source": payment_amount_source, "row_count": len(payment), "paid_user_count": payment["user_id"].nunique()}]
    )
    save_table(source_check, "payment_amount_source_check")
    save_processed(metrics, "03_payment_metrics.csv")
    return metrics, payment[["user_id", "payment_date", "day_offset"]].copy()


def build_vote_metrics(users):
    cell_note(
        "셀 04. 발신 행동축과 수신 반응축 지표",
        "투표 기록을 user_id 기준 발신 행동과 chosen_user_id 기준 수신 발생/열람 반응으로 나누어 집계합니다.",
        "04_sender_vote_metrics.csv, 05_receiver_vote_metrics.csv",
    )
    question = read_csv(
        FILES["question_record"],
        usecols=[
            "id",
            "status",
            "created_at",
            "chosen_user_id",
            "question_id",
            "user_id",
            "has_read",
            "answer_updated_at",
            "opened_times",
        ],
        dtype={
            "id": "string",
            "status": "string",
            "chosen_user_id": "string",
            "question_id": "string",
            "user_id": "string",
        },
    )
    question = question.drop_duplicates(subset=["id"])
    question["user_id"] = as_user_id(question["user_id"])
    question["chosen_user_id"] = as_user_id(question["chosen_user_id"])
    question["created_date"] = to_day(question["created_at"])
    question["answer_updated_date"] = to_day(question["answer_updated_at"])
    question["has_read"] = pd.to_numeric(question["has_read"], errors="coerce").fillna(0).astype(int)
    question["opened_times"] = pd.to_numeric(question["opened_times"], errors="coerce").fillna(0)
    question["is_completed_vote"] = question["status"].fillna("").str.upper().eq("C")
    question = question.dropna(subset=["user_id", "created_date"])

    sent = add_signup_offset(
        question[
            [
                "id",
                "user_id",
                "chosen_user_id",
                "question_id",
                "created_date",
                "is_completed_vote",
            ]
        ].copy(),
        "user_id",
        "created_date",
        users,
    )

    sender_metrics = pd.DataFrame({"user_id": users["user_id"].unique()})
    for name in ["d0", "d0_3", "d0_7", "d0_28"]:
        part = (
            sent.loc[window_mask(sent["day_offset"], name)]
            .groupby("user_id", as_index=False)
            .agg(
                **{
                    f"sent_vote_count_{name}": ("id", "size"),
                    f"sent_vote_complete_count_{name}": ("is_completed_vote", "sum"),
                    f"sent_vote_active_days_{name}": ("created_date", "nunique"),
                    f"sent_vote_distinct_chosen_users_{name}": ("chosen_user_id", "nunique"),
                    f"sent_vote_distinct_questions_{name}": ("question_id", "nunique"),
                }
            )
        )
        part[f"sent_vote_experience_{name}"] = part[f"sent_vote_count_{name}"] > 0
        sender_metrics = sender_metrics.merge(part, on="user_id", how="left")
    sender_metrics = compact_numeric_fill(sender_metrics)
    for c in [c for c in sender_metrics.columns if c.startswith("sent_vote_experience_")]:
        sender_metrics[c] = fill_bool_na_false(sender_metrics[c])
    sender_metrics["sent_vote_complete_rate_d0_7"] = safe_divide(
        sender_metrics["sent_vote_complete_count_d0_7"],
        sender_metrics["sent_vote_count_d0_7"],
    ).clip(0, 1)

    received = question.dropna(subset=["chosen_user_id", "created_date"]).copy()
    received["received_read_flag"] = (received["has_read"] > 0) | (received["opened_times"] > 0)
    received["received_read_date"] = received["answer_updated_date"].fillna(received["created_date"])
    received_occurrence = add_signup_offset(
        received[
            [
                "id",
                "chosen_user_id",
                "created_date",
                "received_read_date",
                "received_read_flag",
                "opened_times",
            ]
        ].copy(),
        "chosen_user_id",
        "created_date",
        users,
    )

    receiver_metrics = pd.DataFrame({"user_id": users["user_id"].unique()})
    for name in ["d0_3", "d0_7", "d0_28"]:
        part = (
            received_occurrence.loc[window_mask(received_occurrence["day_offset"], name)]
            .groupby("user_id", as_index=False)
            .agg(
                **{
                    f"received_vote_count_{name}": ("id", "size"),
                    f"received_vote_days_{name}": ("created_date", "nunique"),
                    f"received_vote_read_count_{name}": ("received_read_flag", "sum"),
                    f"received_vote_opened_times_{name}": ("opened_times", "sum"),
                }
            )
        )
        receiver_metrics = receiver_metrics.merge(part, on="user_id", how="left")

    first_received = (
        received_occurrence.loc[window_mask(received_occurrence["day_offset"], "d0_7")]
        .groupby("user_id", as_index=False)
        .agg(
            first_received_vote_day=("day_offset", "min"),
            first_received_vote_date=("created_date", "min"),
        )
    )
    receiver_metrics = receiver_metrics.merge(first_received, on="user_id", how="left")
    receiver_metrics = compact_numeric_fill(receiver_metrics)
    receiver_metrics["received_vote_yn_d0_7"] = receiver_metrics["received_vote_count_d0_7"] > 0
    receiver_metrics["received_vote_read_rate_d0_7"] = safe_divide(
        receiver_metrics["received_vote_read_count_d0_7"],
        receiver_metrics["received_vote_count_d0_7"],
    ).clip(0, 1)
    receiver_metrics["unread_received_vote_count_d0_7"] = (
        receiver_metrics["received_vote_count_d0_7"] - receiver_metrics["received_vote_read_count_d0_7"]
    ).clip(lower=0)
    receiver_metrics["first_received_vote_day_filled"] = np.where(
        receiver_metrics["received_vote_count_d0_7"] > 0,
        receiver_metrics["first_received_vote_day"],
        8,
    )
    receiver_metrics["early_received_score_d0_7"] = np.where(
        receiver_metrics["received_vote_count_d0_7"] > 0,
        8 - receiver_metrics["first_received_vote_day_filled"],
        0,
    )

    question_sent_activity = sent.loc[
        window_mask(sent["day_offset"], "d0_28"),
        ["user_id", "created_date"],
    ].rename(columns={"created_date": "activity_date"})
    received_read_activity = received_occurrence.loc[
        window_mask(received_occurrence["day_offset"], "d0_28") & received_occurrence["received_read_flag"],
        ["user_id", "received_read_date"],
    ].rename(columns={"received_read_date": "activity_date"})

    save_processed(sender_metrics, "04_sender_vote_metrics.csv")
    save_processed(receiver_metrics, "05_receiver_vote_metrics.csv")
    return sender_metrics, receiver_metrics, question_sent_activity, received_read_activity, first_received


def build_friend_request_metrics(users):
    cell_note(
        "셀 05. 친구 요청/수락 활동",
        "친구 요청 발신과 수락 이벤트를 유저 단위로 집계하고 활동일수 계산용 날짜를 만듭니다.",
        "06_friend_request_metrics.csv",
    )
    metric_parts = []
    active_day_parts = []
    usecols = ["id", "status", "created_at", "updated_at", "receive_user_id", "send_user_id"]
    lookup = users[["user_id", "signup_date"]].drop_duplicates()

    for i, chunk in enumerate(read_csv(FILES["friend_request"], usecols=usecols, chunksize=CHUNKSIZE), start=1):
        chunk = chunk.drop_duplicates(subset=["id"])
        chunk["send_user_id"] = as_user_id(chunk["send_user_id"])
        chunk["receive_user_id"] = as_user_id(chunk["receive_user_id"])
        chunk["created_date"] = to_day(chunk["created_at"])
        chunk["updated_date"] = to_day(chunk["updated_at"])
        chunk["accepted_flag"] = chunk["status"].fillna("").str.lower().isin(["accepted", "accept", "a", "c"])

        sent_chunk = (
            chunk.dropna(subset=["send_user_id", "created_date"])
            .merge(lookup, left_on="send_user_id", right_on="user_id", how="left")
            .drop(columns=["user_id"])
            .rename(columns={"send_user_id": "user_id"})
        )
        sent_chunk["day_offset"] = (sent_chunk["created_date"] - sent_chunk["signup_date"]).dt.days

        recv_chunk = (
            chunk.dropna(subset=["receive_user_id", "created_date"])
            .merge(lookup, left_on="receive_user_id", right_on="user_id", how="left")
            .drop(columns=["user_id"])
            .rename(columns={"receive_user_id": "user_id"})
        )
        recv_chunk["day_offset"] = (recv_chunk["created_date"] - recv_chunk["signup_date"]).dt.days

        for name in ["d0_7", "d0_28"]:
            sent_part = (
                sent_chunk.loc[window_mask(sent_chunk["day_offset"], name)]
                .groupby("user_id", as_index=False)
                .agg(
                    **{
                        f"friend_sent_count_{name}": ("id", "size"),
                        f"friend_sent_accepted_count_{name}": ("accepted_flag", "sum"),
                    }
                )
            )
            recv_part = (
                recv_chunk.loc[window_mask(recv_chunk["day_offset"], name)]
                .groupby("user_id", as_index=False)
                .agg(
                    **{
                        f"friend_received_count_{name}": ("id", "size"),
                        f"friend_received_accepted_count_{name}": ("accepted_flag", "sum"),
                    }
                )
            )
            metric_parts.append(sent_part.merge(recv_part, on="user_id", how="outer"))

        active_sent = sent_chunk.loc[window_mask(sent_chunk["day_offset"], "d0_28"), ["user_id", "created_date"]]
        active_sent = active_sent.rename(columns={"created_date": "activity_date"})
        recv_chunk["accept_day_offset"] = (recv_chunk["updated_date"] - recv_chunk["signup_date"]).dt.days
        active_accept = recv_chunk.loc[
            recv_chunk["accepted_flag"] & window_mask(recv_chunk["accept_day_offset"], "d0_28"),
            ["user_id", "updated_date"],
        ].rename(columns={"updated_date": "activity_date"})
        active_day_parts.append(pd.concat([active_sent, active_accept], ignore_index=True).drop_duplicates())

        if i % 5 == 0:
            print(f"friend_request processed chunks: {i}")

    metrics = compact_numeric_fill(concat_grouped_sum(metric_parts, users))
    activity = (
        pd.concat(active_day_parts, ignore_index=True).drop_duplicates()
        if active_day_parts
        else pd.DataFrame(columns=["user_id", "activity_date"])
    )
    save_processed(metrics, "06_friend_request_metrics.csv")
    return metrics, activity


def build_point_metrics(users):
    cell_note(
        "셀 06. 포인트 사용 활동",
        "포인트 감소 이력을 포인트 사용 행동으로 보고 활동일수 및 수신 이후 소비 반응 지표에 사용할 날짜를 만듭니다.",
        "07_point_metrics.csv",
    )
    activity_parts = []
    metric_parts = []
    usecols = ["id", "delta_point", "created_at", "user_id"]
    lookup = users[["user_id", "signup_date"]].drop_duplicates()

    for i, chunk in enumerate(read_csv(FILES["point_history"], usecols=usecols, chunksize=CHUNKSIZE), start=1):
        chunk = chunk.drop_duplicates(subset=["id"])
        chunk["user_id"] = as_user_id(chunk["user_id"])
        chunk["point_date"] = to_day(chunk["created_at"])
        chunk["delta_point"] = pd.to_numeric(chunk["delta_point"], errors="coerce").fillna(0)
        chunk = chunk.dropna(subset=["user_id", "point_date"]).merge(lookup, on="user_id", how="left")
        chunk["day_offset"] = (chunk["point_date"] - chunk["signup_date"]).dt.days
        chunk["point_use_flag"] = chunk["delta_point"] < 0

        activity_parts.append(
            chunk.loc[window_mask(chunk["day_offset"], "d0_28") & chunk["point_use_flag"], ["user_id", "point_date"]]
            .rename(columns={"point_date": "activity_date"})
            .drop_duplicates()
        )
        for name in ["d0_7", "d0_28"]:
            metric_parts.append(
                chunk.loc[window_mask(chunk["day_offset"], name)]
                .groupby("user_id", as_index=False)
                .agg(
                    **{
                        f"point_event_count_{name}": ("id", "size"),
                        f"point_use_count_{name}": ("point_use_flag", "sum"),
                        f"point_delta_sum_{name}": ("delta_point", "sum"),
                    }
                )
            )

        if i % 5 == 0:
            print(f"point_history processed chunks: {i}")

    metrics = compact_numeric_fill(concat_grouped_sum(metric_parts, users))
    activity = (
        pd.concat(activity_parts, ignore_index=True).drop_duplicates()
        if activity_parts
        else pd.DataFrame(columns=["user_id", "activity_date"])
    )
    save_processed(metrics, "07_point_metrics.csv")
    return metrics, activity


def build_active_day_and_network_metrics(users, activity_frames):
    cell_note(
        "셀 07. 누적 활동일수와 관계망 활성도",
        "발신, 수신 열람, 친구 요청/수락, 포인트 사용 활동일수를 만들고 같은 반/학년 활성 유저 수를 계산합니다.",
        "08_active_day_metrics.csv, 09_network_active_metrics.csv",
    )
    activity_days = pd.concat(activity_frames, ignore_index=True).dropna(subset=["user_id", "activity_date"])
    activity_days["user_id"] = as_user_id(activity_days["user_id"])
    activity_days["activity_date"] = pd.to_datetime(activity_days["activity_date"], errors="coerce").dt.floor("D")
    activity_days = activity_days.dropna(subset=["user_id", "activity_date"]).drop_duplicates()
    activity_days = add_signup_offset(activity_days, "user_id", "activity_date", users)

    active_day_metrics = pd.DataFrame({"user_id": users["user_id"].unique()})
    for name in ["d0_7", "d0_28", "d8_28"]:
        part = (
            activity_days.loc[window_mask(activity_days["day_offset"], name)]
            .groupby("user_id", as_index=False)
            .agg(**{f"active_days_{name}": ("activity_date", "nunique")})
        )
        active_day_metrics = active_day_metrics.merge(part, on="user_id", how="left")
    active_day_metrics = compact_numeric_fill(active_day_metrics)

    active_users_d0_7 = activity_days.loc[window_mask(activity_days["day_offset"], "d0_7"), "user_id"].drop_duplicates()
    user_active = users[["user_id", "class_key", "grade_key", "same_class_total_users", "same_grade_total_users"]].copy()
    user_active["is_active_d0_7"] = user_active["user_id"].isin(set(active_users_d0_7.astype("string")))

    active_by_class = (
        user_active.loc[user_active["is_active_d0_7"] & user_active["class_key"].notna()]
        .groupby("class_key", as_index=False)
        .agg(class_active_users_d0_7=("user_id", "nunique"))
    )
    active_by_grade = (
        user_active.loc[user_active["is_active_d0_7"] & user_active["grade_key"].notna()]
        .groupby("grade_key", as_index=False)
        .agg(grade_active_users_d0_7=("user_id", "nunique"))
    )
    network_active = (
        user_active.merge(active_by_class, on="class_key", how="left")
        .merge(active_by_grade, on="grade_key", how="left")
    )
    network_active["class_active_users_d0_7"] = network_active["class_active_users_d0_7"].fillna(0)
    network_active["grade_active_users_d0_7"] = network_active["grade_active_users_d0_7"].fillna(0)
    network_active["same_class_active_d0_7"] = (
        network_active["class_active_users_d0_7"] - network_active["is_active_d0_7"].astype(int)
    ).clip(lower=0)
    network_active["same_grade_active_d0_7"] = (
        network_active["grade_active_users_d0_7"] - network_active["is_active_d0_7"].astype(int)
    ).clip(lower=0)
    network_active["same_class_active_ratio_d0_7"] = safe_divide(
        network_active["same_class_active_d0_7"],
        (network_active["same_class_total_users"] - 1).clip(lower=0),
    ).clip(0, 1)
    network_active["same_grade_active_ratio_d0_7"] = safe_divide(
        network_active["same_grade_active_d0_7"],
        (network_active["same_grade_total_users"] - 1).clip(lower=0),
    ).clip(0, 1)

    network_cols = [
        "user_id",
        "same_class_active_d0_7",
        "same_grade_active_d0_7",
        "same_class_active_ratio_d0_7",
        "same_grade_active_ratio_d0_7",
    ]
    network_metrics = network_active[network_cols].copy()

    save_processed(active_day_metrics, "08_active_day_metrics.csv")
    save_processed(network_metrics, "09_network_active_metrics.csv")
    return active_day_metrics, network_metrics, activity_days


def build_hackle_metrics(users):
    cell_note(
        "셀 08. Hackle 질문 행동 보완 지표",
        "session_id를 user_id로 연결해 D0-D7 질문 시작/완료/스킵 행동을 발신축 보완 변수로 집계합니다.",
        "10_hackle_question_metrics.csv",
    )
    if not RUN_HACKLE_SUPPLEMENT:
        return pd.DataFrame({"user_id": users["user_id"].unique()})
    if not (FILES["hackle_properties"].exists() and FILES["hackle_events"].exists()):
        print("[WARN] Hackle files are missing. Skipping Hackle supplement.")
        return pd.DataFrame({"user_id": users["user_id"].unique()})

    hprop = read_csv(
        FILES["hackle_properties"],
        usecols=["session_id", "user_id"],
        dtype={"session_id": "string", "user_id": "string"},
    )
    hprop["user_id"] = as_user_id(hprop["user_id"])
    hprop = hprop.dropna(subset=["session_id", "user_id"]).drop_duplicates(subset=["session_id"])
    lookup = users[["user_id", "signup_date"]].drop_duplicates()

    selected_keys = sorted(set(HACKLE_QUESTION_START_KEYS + HACKLE_QUESTION_COMPLETE_KEYS + HACKLE_QUESTION_SKIP_KEYS))
    metric_parts = []
    usecols = ["event_id", "event_datetime", "event_key", "session_id"]

    for i, chunk in enumerate(read_csv(FILES["hackle_events"], usecols=usecols, chunksize=CHUNKSIZE), start=1):
        chunk["event_key"] = chunk["event_key"].astype("string")
        chunk = chunk[chunk["event_key"].isin(selected_keys)].copy()
        if chunk.empty:
            continue
        chunk["event_date"] = to_day(chunk["event_datetime"])
        chunk = chunk.dropna(subset=["event_date", "session_id"])
        chunk = chunk.merge(hprop, on="session_id", how="inner").merge(lookup, on="user_id", how="left")
        chunk["day_offset"] = (chunk["event_date"] - chunk["signup_date"]).dt.days
        chunk = chunk[window_mask(chunk["day_offset"], "d0_7")]
        if chunk.empty:
            continue
        chunk["hackle_question_start"] = chunk["event_key"].isin(HACKLE_QUESTION_START_KEYS)
        chunk["hackle_question_complete"] = chunk["event_key"].isin(HACKLE_QUESTION_COMPLETE_KEYS)
        chunk["hackle_question_skip"] = chunk["event_key"].isin(HACKLE_QUESTION_SKIP_KEYS)
        metric_parts.append(
            chunk.groupby("user_id", as_index=False).agg(
                hackle_question_start_count_d0_7=("hackle_question_start", "sum"),
                hackle_question_complete_count_d0_7=("hackle_question_complete", "sum"),
                hackle_question_skip_count_d0_7=("hackle_question_skip", "sum"),
                hackle_question_active_days_d0_7=("event_date", "nunique"),
            )
        )
        if i % 5 == 0:
            print(f"hackle_events processed chunks: {i}")

    metrics = compact_numeric_fill(concat_grouped_sum(metric_parts, users))
    for c in [
        "hackle_question_start_count_d0_7",
        "hackle_question_complete_count_d0_7",
        "hackle_question_skip_count_d0_7",
        "hackle_question_active_days_d0_7",
    ]:
        if c not in metrics.columns:
            metrics[c] = 0
    metrics["hackle_question_completion_rate_d0_7"] = safe_divide(
        metrics["hackle_question_complete_count_d0_7"],
        metrics["hackle_question_start_count_d0_7"],
    ).clip(0, 1)
    metrics["hackle_question_skip_rate_d0_7"] = safe_divide(
        metrics["hackle_question_skip_count_d0_7"],
        metrics["hackle_question_start_count_d0_7"],
    ).clip(0, 1)
    save_processed(metrics, "10_hackle_question_metrics.csv")
    return metrics


def build_candidate_exposure_metrics(users, receiver_metrics):
    cell_note(
        "셀 09. 후보 노출 지표",
        "polls_usercandidate 기준 후보 노출 횟수와 노출 대비 선택률을 관계망 환경축 변수로 만듭니다.",
        "11_candidate_exposure_metrics.csv",
    )
    if not FILES["candidate_exposure"].exists():
        print("[WARN] candidate exposure file is missing. Skipping.")
        return pd.DataFrame({"user_id": users["user_id"].unique()})

    question_piece_lookup = None
    if FILES["question_piece"].exists():
        question_piece_lookup = read_csv(
            FILES["question_piece"],
            usecols=["id", "question_id"],
            dtype={"id": "string", "question_id": "string"},
        ).drop_duplicates(subset=["id"])
        question_piece_lookup = question_piece_lookup.rename(columns={"id": "question_piece_id"})

    lookup = users[["user_id", "signup_date"]].drop_duplicates()
    parts = []
    usecols = ["id", "created_at", "question_piece_id", "user_id"]
    for i, chunk in enumerate(read_csv(FILES["candidate_exposure"], usecols=usecols, chunksize=CHUNKSIZE), start=1):
        chunk = chunk.drop_duplicates(subset=["id"])
        chunk["user_id"] = as_user_id(chunk["user_id"])
        chunk["question_piece_id"] = chunk["question_piece_id"].astype("string")
        chunk["candidate_exposure_date"] = to_day(chunk["created_at"])
        chunk = chunk.dropna(subset=["user_id", "candidate_exposure_date"])
        if question_piece_lookup is not None:
            chunk = chunk.merge(question_piece_lookup, on="question_piece_id", how="left")
            chunk["question_id"] = chunk["question_id"].fillna(chunk["question_piece_id"])
        else:
            chunk["question_id"] = chunk["question_piece_id"]
        chunk = chunk.merge(lookup, on="user_id", how="left")
        chunk["day_offset"] = (chunk["candidate_exposure_date"] - chunk["signup_date"]).dt.days
        chunk = chunk.loc[window_mask(chunk["day_offset"], "d0_7")]
        if chunk.empty:
            continue
        parts.append(
            chunk.groupby("user_id", as_index=False).agg(
                candidate_exposure_count_d0_7=("id", "size"),
                candidate_unique_question_count_d0_7=("question_id", "nunique"),
                candidate_exposure_days_d0_7=("candidate_exposure_date", "nunique"),
            )
        )
        if i % 5 == 0:
            print(f"candidate exposure processed chunks: {i}")

    metrics = compact_numeric_fill(concat_grouped_sum(parts, users))
    metrics = metrics.merge(
        receiver_metrics[["user_id", "received_vote_count_d0_7"]],
        on="user_id",
        how="left",
    )
    metrics["received_vote_count_d0_7"] = pd.to_numeric(metrics["received_vote_count_d0_7"], errors="coerce").fillna(0)
    metrics["candidate_to_recv_rate_d0_7"] = safe_divide(
        metrics["received_vote_count_d0_7"],
        metrics["candidate_exposure_count_d0_7"],
    ).clip(0, 1)
    metrics = metrics.drop(columns=["received_vote_count_d0_7"], errors="ignore")
    save_processed(metrics, "11_candidate_exposure_metrics.csv")
    return metrics


def build_after_received_metrics(first_received, attendance, point_activity, payment, users):
    cell_note(
        "셀 10. 수신 이후 반응 지표",
        "첫 수신 투표 이후 출석, 포인트 사용, 결제가 관찰되는지 시간적 연결성 지표를 만듭니다.",
        "12_after_received_metrics.csv",
    )
    if first_received.empty:
        return pd.DataFrame(columns=["user_id"])

    first = first_received[["user_id", "first_received_vote_date"]].dropna().copy()

    att = attendance.merge(first, on="user_id", how="inner")
    att_after = att.loc[
        (att["attendance_date"] >= att["first_received_vote_date"]) & window_mask(att["day_offset"], "d0_28")
    ]
    att_metrics = (
        att_after.groupby("user_id", as_index=False)
        .agg(attendance_days_after_received_vote_d0_28=("attendance_date", "nunique"))
    )

    point = point_activity.copy()
    point["activity_date"] = to_day(point["activity_date"])
    point = add_signup_offset(point.dropna(subset=["user_id", "activity_date"]), "user_id", "activity_date", users)
    point = point.merge(first, on="user_id", how="inner", suffixes=("", "_first"))
    point_after = point.loc[
        (point["activity_date"] >= point["first_received_vote_date"]) & window_mask(point["day_offset"], "d0_28")
    ]
    point_metrics = (
        point_after.groupby("user_id", as_index=False)
        .agg(point_use_after_received_vote_count_d0_28=("activity_date", "size"))
    )

    pay = payment.merge(first, on="user_id", how="inner")
    pay_after = pay.loc[
        (pay["payment_date"] >= pay["first_received_vote_date"]) & window_mask(pay["day_offset"], "d0_28")
    ]
    pay_metrics = (
        pay_after.groupby("user_id", as_index=False)
        .agg(payment_after_received_vote_count_d0_28=("payment_date", "size"))
    )

    out = first[["user_id"]].drop_duplicates()
    for frame in [att_metrics, point_metrics, pay_metrics]:
        out = out.merge(frame, on="user_id", how="left")
    out = compact_numeric_fill(out)
    out["attendance_after_received_vote_yn_d0_28"] = out["attendance_days_after_received_vote_d0_28"] > 0
    out["point_use_after_received_vote_yn_d0_28"] = out["point_use_after_received_vote_count_d0_28"] > 0
    out["payment_after_received_vote_yn_d0_28"] = out["payment_after_received_vote_count_d0_28"] > 0
    save_processed(out, "12_after_received_metrics.csv")
    return out


# %% [markdown]
# ## 셀3-4. 유저 단위 분석 테이블 생성 함수


# %%
def prepare_analysis_table(
    users,
    attendance_metrics,
    payment_metrics,
    sender_metrics,
    receiver_metrics,
    friend_metrics,
    point_metrics,
    active_day_metrics,
    network_metrics,
    hackle_metrics,
    candidate_metrics,
    after_received_metrics,
):
    cell_note(
        "셀 11. 3축 유저 단위 분석 테이블",
        "발신 행동축, 수신 반응축, 관계망 환경축, 결과 지표를 user_id 기준 1행 테이블로 병합합니다.",
        "13_three_axis_user_level_analysis.csv",
    )
    base_cols = [
        "user_id",
        "signup_at",
        "signup_date",
        "friend_count",
        "pending_votes",
        "group_id",
        "school_id",
        "grade",
        "class_num",
        "same_school_joined_user_count",
        "same_class_total_users",
        "same_grade_total_users",
        "home_entry_population",
    ]
    analysis = users[base_cols].copy()
    for frame in [
        attendance_metrics,
        payment_metrics,
        sender_metrics,
        receiver_metrics,
        friend_metrics,
        point_metrics,
        active_day_metrics,
        network_metrics,
        hackle_metrics,
        candidate_metrics,
        after_received_metrics,
    ]:
        if frame is not None and not frame.empty:
            analysis = analysis.merge(frame, on="user_id", how="left")

    analysis = compact_numeric_fill(analysis)
    bool_cols = [
        "home_entry_population",
        "retention_d1_28",
        "retention_d8_28",
        "paid_flag_d0_28",
        "paid_flag_d8_28",
        "sent_vote_experience_d0_7",
        "received_vote_yn_d0_7",
        "attendance_after_received_vote_yn_d0_28",
        "point_use_after_received_vote_yn_d0_28",
        "payment_after_received_vote_yn_d0_28",
    ]
    for col in bool_cols:
        if col in analysis.columns:
            analysis[col] = fill_bool_na_false(analysis[col])

    for col in SENDER_FEATURES + RECEIVER_FEATURES + NETWORK_FEATURES + PROFILE_ONLY_FEATURES + OUTCOME_COLUMNS:
        if col not in analysis.columns:
            analysis[col] = 0

    analysis["sender_group"] = np.where(analysis["sent_vote_count_d0_7"] > 0, "보낸 투표 경험 유", "보낸 투표 경험 무")
    analysis["receiver_group"] = np.where(analysis["received_vote_count_d0_7"] > 0, "받은 투표 경험 유", "받은 투표 경험 무")

    save_processed(analysis, "13_three_axis_user_level_analysis.csv")
    return analysis


# %% [markdown]
# ## 셀3-5. 클러스터링 입력 변수 정의 및 표준화 함수


# %%
def build_feature_definition(available_features):
    rows = []
    for axis, features, note in [
        ("sender_behavior", SENDER_FEATURES, "Final_KimSuHyun.ipynb 발신 행동축: D0-D7 능동 투표 행동"),
        ("receiver_reaction", RECEIVER_FEATURES, "수신_반응축.pdf: 수신 발생/강도/열람 반응"),
        ("network_environment", NETWORK_FEATURES, "관계망_환경축.pdf: 친구 수, 같은 반/학년 활성도, 후보 노출"),
    ]:
        for feature in features:
            rows.append(
                {
                    "axis": axis,
                    "feature": feature,
                    "feature_label": label_for(feature),
                    "use_in_clustering": feature in available_features,
                    "source_note": note,
                }
            )
    for feature in PROFILE_ONLY_FEATURES:
        rows.append(
            {
                "axis": "network_environment",
                "feature": feature,
                "feature_label": label_for(feature),
                "use_in_clustering": False,
                "source_note": "프로파일 해석용 보조 변수",
            }
        )
    return pd.DataFrame(rows)


def add_segment_buckets(df):
    out = df.copy()
    out["sender_intensity_bucket_d0_7"] = pd.cut(
        pd.to_numeric(out["sent_vote_count_d0_7"], errors="coerce").fillna(0),
        bins=[-0.1, 0, 3, 10, 50, np.inf],
        labels=["0회", "1-3회", "4-10회", "11-50회", "51회 이상"],
        ordered=True,
    )
    out["receiver_intensity_bucket_d0_7"] = pd.cut(
        pd.to_numeric(out["received_vote_count_d0_7"], errors="coerce").fillna(0),
        bins=[-0.1, 0, 1, 3, 10, np.inf],
        labels=["0회", "1회", "2-3회", "4-10회", "11회 이상"],
        ordered=True,
    )
    out["candidate_exposure_bucket_d0_7"] = pd.cut(
        pd.to_numeric(out["candidate_exposure_count_d0_7"], errors="coerce").fillna(0),
        bins=[-0.1, 0, 5, 20, np.inf],
        labels=["노출 없음", "1-5회", "6-20회", "21회 이상"],
        ordered=True,
    )
    out["friend_count_bucket"] = pd.cut(
        pd.to_numeric(out["friend_count"], errors="coerce").fillna(0),
        bins=[3.999, 10, 30, 70, 150, np.inf],
        labels=["4-10명", "11-30명", "31-70명", "71-150명", "151명 이상"],
        ordered=True,
    )
    return out


def transform_cluster_features(cluster_base, features):
    raw = cluster_base[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    transformed = raw.copy()
    transform_rows = []

    rate_cols = [c for c in features if c.endswith("_rate_d0_7") or c.endswith("_ratio_d0_7")]
    bounded_cols = rate_cols + ["sent_vote_complete_rate_d0_7", "received_vote_read_rate_d0_7"]
    timing_score_cols = ["early_received_score_d0_7"]
    count_cols = [c for c in features if c not in set(bounded_cols + timing_score_cols)]

    for col in count_cols:
        transformed[col] = np.log1p(winsorize_series(raw[col], 0.99))
        transform_rows.append(
            {
                "feature": col,
                "feature_label": label_for(col),
                "transform": "p99 winsorize -> log1p -> StandardScaler",
                "p99": float(pd.to_numeric(raw[col], errors="coerce").fillna(0).quantile(0.99)),
            }
        )
    for col in bounded_cols:
        if col in transformed.columns:
            transformed[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).clip(0, 1)
            transform_rows.append(
                {
                    "feature": col,
                    "feature_label": label_for(col),
                    "transform": "clip 0-1 -> StandardScaler",
                    "p99": float(pd.to_numeric(raw[col], errors="coerce").fillna(0).quantile(0.99)),
                }
            )
    for col in timing_score_cols:
        if col in transformed.columns:
            transformed[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).clip(0, 8)
            transform_rows.append(
                {
                    "feature": col,
                    "feature_label": label_for(col),
                    "transform": "clip 0-8 -> StandardScaler; 0 means no D0-D7 receipt, 8 means D0 receipt",
                    "p99": float(pd.to_numeric(raw[col], errors="coerce").fillna(0).quantile(0.99)),
                }
            )

    non_constant = [c for c in transformed.columns if transformed[c].std(ddof=0) > 0]
    if len(non_constant) < 2:
        raise ValueError("클러스터링에 사용할 변동성 있는 변수가 2개 미만입니다.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed[non_constant])
    scaled_df = pd.DataFrame(scaled, columns=non_constant, index=cluster_base.index)
    return raw, transformed[non_constant], scaled_df, pd.DataFrame(transform_rows)


# %% [markdown]
# ## 셀3-6. k 선택 및 클러스터링 보조 함수


# %%
def add_elbow_metrics(metrics):
    def minmax_score(series):
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        min_value = s.min()
        max_value = s.max()
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - min_value) / (max_value - min_value)

    metrics = metrics.sort_values("k").reset_index(drop=True)
    prev_inertia = metrics["inertia"].shift(1)
    metrics["inertia_drop_from_prev"] = (prev_inertia - metrics["inertia"]).fillna(0).round(3)
    metrics["inertia_drop_pct_from_prev"] = (
        metrics["inertia_drop_from_prev"] / prev_inertia.replace(0, np.nan) * 100
    ).fillna(0).round(2)

    if len(metrics) >= 3 and metrics["inertia"].nunique() > 1 and metrics["k"].nunique() > 1:
        x = (metrics["k"] - metrics["k"].min()) / (metrics["k"].max() - metrics["k"].min())
        y = (metrics["inertia"] - metrics["inertia"].min()) / (metrics["inertia"].max() - metrics["inertia"].min())
        metrics["elbow_score"] = (x + y - 1).abs() / np.sqrt(2)
        metrics["elbow_score"] = metrics["elbow_score"].round(4)
    else:
        metrics["elbow_score"] = 0.0

    valid_elbow = metrics.loc[metrics["meets_min_cluster_share"] & metrics["k"].ge(3)].copy()
    if valid_elbow.empty:
        valid_elbow = metrics.loc[metrics["meets_min_cluster_share"]].copy()
    if valid_elbow.empty:
        valid_elbow = metrics.copy()
    elbow_k = int(valid_elbow.sort_values(["elbow_score", "k"], ascending=[False, True]).iloc[0]["k"])
    metrics["is_elbow_k"] = metrics["k"].eq(elbow_k)
    metrics["elbow_score_norm"] = minmax_score(metrics["elbow_score"]).round(4)
    metrics["silhouette_score_norm"] = minmax_score(metrics["silhouette_score"]).round(4)
    metrics["inertia_drop_pct_norm"] = minmax_score(metrics["inertia_drop_pct_from_prev"]).round(4)
    metrics["auto_k_score"] = (
        metrics["elbow_score_norm"] * AUTO_K_ELBOW_WEIGHT
        + metrics["silhouette_score_norm"] * AUTO_K_SILHOUETTE_WEIGHT
        + metrics["inertia_drop_pct_norm"] * AUTO_K_DROP_WEIGHT
    ).round(4)
    return metrics


def suggest_k_from_metrics(metrics):
    valid = metrics.loc[metrics["meets_min_cluster_share"]].copy()
    if valid.empty:
        valid = metrics.copy()
    preferred = valid.loc[
        valid["k"].between(PREFERRED_AUTO_K_RANGE[0], PREFERRED_AUTO_K_RANGE[1], inclusive="both")
    ].copy()
    candidate = preferred if not preferred.empty else valid
    best_row = candidate.sort_values(
        ["auto_k_score", "elbow_score", "silhouette_score", "min_cluster_share_pct", "k"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    best_k = int(best_row["k"])
    reason = (
        "elbow score, inertia 감소율, 실루엣 점수를 가중 합산한 auto_k_score가 가장 높은 k를 자동 선택했습니다. "
        f"가중치: elbow {AUTO_K_ELBOW_WEIGHT:.2f}, silhouette {AUTO_K_SILHOUETTE_WEIGHT:.2f}, "
        f"inertia drop {AUTO_K_DROP_WEIGHT:.2f}."
    )
    return best_k, reason


def save_k_selection_figure(k_metrics, selected_k=None):
    value_cols = ["inertia", "elbow_score", "silhouette_score", "auto_k_score", "min_cluster_share_pct"]
    validate_graph_data(
        "three_axis_cluster_k_selection",
        k_metrics,
        required_cols=["k"] + value_cols,
        value_cols=value_cols,
        allow_all_zero=False,
    )
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5))
    line_specs = [
        ("inertia", "군집 내 제곱합", "inertia", "#2E5EAA"),
        ("elbow_score", "Elbow score", "elbow score", "#9A6A3A"),
        ("silhouette_score", "실루엣 점수", "sampled silhouette", "#3B8F63"),
        ("auto_k_score", "자동 k 선택 점수", "weighted score", "#C05A5A"),
        ("min_cluster_share_pct", "최소 군집 비율", "%", "#7B5EA7"),
    ]
    for ax, (metric, title, ylabel, color) in zip(axes, line_specs):
        sns.lineplot(data=k_metrics, x="k", y=metric, marker="o", ax=ax, color=color)
        if selected_k is not None:
            ax.axvline(selected_k, color="#C93A3A", linestyle="--", linewidth=1)
        if "is_elbow_k" in k_metrics.columns:
            elbow_rows = k_metrics.loc[k_metrics["is_elbow_k"].eq(True)]
        else:
            elbow_rows = pd.DataFrame()
        if not elbow_rows.empty:
            ax.axvline(int(elbow_rows.iloc[0]["k"]), color="#555555", linestyle=":", linewidth=1)
        if metric == "min_cluster_share_pct":
            ax.axhline(MIN_CLUSTER_SHARE * 100, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
    save_fig(fig, "three_axis_cluster_k_selection")


def choose_cluster_k(X_scaled):
    n_rows = X_scaled.shape[0]
    unique_rows = pd.DataFrame(X_scaled).drop_duplicates().shape[0]
    max_k = min(MAX_THREE_AXIS_CLUSTER_K, n_rows - 1, unique_rows)
    if max_k < 2:
        raise ValueError("클러스터링 가능한 고유 행이 부족합니다.")

    k_values = list(range(2, max_k + 1))
    rows = []
    fitted_labels = {}
    for k in k_values:
        model = build_model(k, n_rows)
        labels = model.fit_predict(X_scaled)
        fitted_labels[k] = labels
        counts = pd.Series(labels).value_counts().sort_index()
        min_share = counts.min() / len(labels)
        sample_idx = sampled_indices(n_rows, SILHOUETTE_SAMPLE_SIZE, RANDOM_STATE + k)
        sample_labels = labels[sample_idx]
        if len(np.unique(sample_labels)) >= 2:
            sil = silhouette_score(X_scaled[sample_idx], sample_labels)
        else:
            sil = np.nan
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": sil,
                "min_cluster_size": int(counts.min()),
                "min_cluster_share_pct": round(min_share * 100, 2),
                "meets_min_cluster_share": bool(min_share >= MIN_CLUSTER_SHARE),
                "cluster_size_summary": ", ".join(
                    f"{cluster_id}: {count:,}명 ({count / len(labels) * 100:.2f}%)"
                    for cluster_id, count in counts.items()
                ),
                "silhouette_sample_size": int(len(sample_idx)),
                "model_type": type(model).__name__,
            }
        )
        print(f"k={k}, silhouette={sil:.4f}, min_share={min_share:.2%}")

    metrics = add_elbow_metrics(pd.DataFrame(rows))
    suggested_k, suggestion_reason = suggest_k_from_metrics(metrics)
    metrics["auto_suggested_k"] = metrics["k"].eq(suggested_k)
    metrics["auto_suggestion_reason"] = np.where(metrics["auto_suggested_k"], suggestion_reason, "")

    if SELECTED_THREE_AXIS_K is not None:
        if SELECTED_THREE_AXIS_K not in k_values:
            raise ValueError(f"SELECTED_THREE_AXIS_K={SELECTED_THREE_AXIS_K} is not in candidate k values {k_values}.")
        best_k = int(SELECTED_THREE_AXIS_K)
        reason = "그래프 확인 후 수동 지정한 k를 사용했습니다."
    else:
        if REQUIRE_MANUAL_K_SELECTION:
            save_table(metrics, "three_axis_cluster_k_selection_metrics")
            save_k_selection_figure(metrics, selected_k=suggested_k)
            if FIGURE_CAPTION_ROWS:
                save_table(pd.DataFrame(FIGURE_CAPTION_ROWS), "three_axis_figure_interpretations")
            if GRAPH_VALIDATION_ROWS:
                save_table(pd.DataFrame(GRAPH_VALIDATION_ROWS), "three_axis_graph_data_validation")
            raise ValueError(
                "k 후보 그래프와 표를 저장했습니다. "
                f"그래프를 확인한 뒤 SELECTED_THREE_AXIS_K에 원하는 k를 입력하고 다시 실행하세요. "
                f"참고용 자동 제안 k는 {suggested_k}입니다."
            )
        best_k = suggested_k
        reason = suggestion_reason

    metrics["selected_k"] = metrics["k"].eq(best_k)
    metrics["selection_reason"] = np.where(metrics["selected_k"], reason, "")
    return best_k, metrics, fitted_labels.get(best_k)


def add_axis_scores(cluster_base, scaled_df):
    axis_map = {
        "sender_axis_score": [c for c in SENDER_FEATURES if c in scaled_df.columns],
        "receiver_axis_score": [c for c in RECEIVER_FEATURES if c in scaled_df.columns],
        "network_axis_score": [c for c in NETWORK_FEATURES if c in scaled_df.columns],
    }
    for score_col, cols in axis_map.items():
        cluster_base[score_col] = scaled_df[cols].mean(axis=1) if cols else 0
    return cluster_base


def build_cluster_labels(cluster_base):
    axis_profile = (
        cluster_base.groupby("cluster", as_index=False)
        .agg(
            user_count=("user_id", "nunique"),
            sender_axis_score=("sender_axis_score", "mean"),
            receiver_axis_score=("receiver_axis_score", "mean"),
            network_axis_score=("network_axis_score", "mean"),
            sent_vote_count_d0_7=("sent_vote_count_d0_7", "mean"),
            received_vote_count_d0_7=("received_vote_count_d0_7", "mean"),
            friend_count=("friend_count", "mean"),
            same_class_active_d0_7=("same_class_active_d0_7", "mean"),
            same_grade_active_d0_7=("same_grade_active_d0_7", "mean"),
            candidate_exposure_count_d0_7=("candidate_exposure_count_d0_7", "mean"),
            pending_votes=("pending_votes", "mean"),
        )
    )

    axis_profile["activity_score"] = axis_profile[["sender_axis_score", "receiver_axis_score"]].max(axis=1)
    relationship_cols = ["friend_count", "same_class_active_d0_7", "same_grade_active_d0_7"]
    axis_profile["relationship_score"] = axis_profile[relationship_cols].rank(pct=True).mean(axis=1)

    activity_threshold = max(0.25, axis_profile["activity_score"].median())
    relationship_threshold = axis_profile["relationship_score"].median()
    pending_low_threshold = axis_profile["pending_votes"].quantile(1 / 3)
    pending_high_threshold = axis_profile["pending_votes"].quantile(2 / 3)

    def pending_level(value):
        if value <= pending_low_threshold:
            return "저적체"
        if value >= pending_high_threshold:
            return "고적체"
        return "중적체"

    label_map = {}
    rule_map = {}
    for _, row in axis_profile.iterrows():
        cid = int(row["cluster"])
        activity_high = row["activity_score"] >= activity_threshold
        relationship_high = row["relationship_score"] >= relationship_threshold
        activity_part = "고활동" if activity_high else "저활동"
        relationship_part = "고관계" if relationship_high else "저관계"
        pending_part = pending_level(row["pending_votes"])
        label = f"{activity_part}·{relationship_part}·{pending_part} (n={int(row['user_count']):,})"
        rule = (
            f"초기 발신/수신 활동은 {activity_part[:1]}수준, 관계망은 {relationship_part[:1]}수준이며 "
            f"미확인 Ping 평균은 {pending_part[:1]}수준인 군집"
        )

        label_map[cid] = label
        rule_map[cid] = rule

    label_map = unique_labels(label_map)
    return label_map, rule_map, axis_profile


# %% [markdown]
# ## 셀3-7. 클러스터별 결과표 및 구간별 결과표 함수


# %%
def outcome_aggregation_dict():
    return {
        "user_count": ("user_id", "nunique"),
        "retention_d1_28_rate": ("retention_d1_28", rate),
        "retention_d8_28_rate": ("retention_d8_28", rate),
        "payment_d0_28_rate": ("paid_flag_d0_28", rate),
        "payment_d8_28_rate": ("paid_flag_d8_28", rate),
        "payment_d0_28_count_mean": ("payment_count_d0_28", "mean"),
        "payment_d0_28_count_sum": ("payment_count_d0_28", "sum"),
        "payment_d0_28_paid_user_count": ("paid_flag_d0_28", "sum"),
        "payment_d8_28_count_mean": ("payment_count_d8_28", "mean"),
        "payment_d8_28_count_sum": ("payment_count_d8_28", "sum"),
        "payment_d8_28_paid_user_count": ("paid_flag_d8_28", "sum"),
        "payment_d0_28_amount_mean": ("payment_amount_d0_28", "mean"),
        "payment_d0_28_amount_sum": ("payment_amount_d0_28", "sum"),
        "payment_d8_28_amount_mean": ("payment_amount_d8_28", "mean"),
        "payment_d8_28_amount_sum": ("payment_amount_d8_28", "sum"),
        "active_days_d0_28_mean": ("active_days_d0_28", "mean"),
        "active_days_d0_28_sum": ("active_days_d0_28", "sum"),
        "active_days_d0_28_active_user_count": ("active_days_d0_28", nonzero_count),
        "active_days_d8_28_mean": ("active_days_d8_28", "mean"),
        "active_days_d8_28_sum": ("active_days_d8_28", "sum"),
        "active_days_d8_28_active_user_count": ("active_days_d8_28", nonzero_count),
        "sent_vote_count_d0_7_mean": ("sent_vote_count_d0_7", "mean"),
        "sent_vote_count_d0_7_sum": ("sent_vote_count_d0_7", "sum"),
        "sent_vote_user_count_d0_7": ("sent_vote_count_d0_7", nonzero_count),
        "received_vote_count_d0_7_mean": ("received_vote_count_d0_7", "mean"),
        "received_vote_count_d0_7_sum": ("received_vote_count_d0_7", "sum"),
        "received_vote_user_count_d0_7": ("received_vote_count_d0_7", nonzero_count),
        "candidate_exposure_count_d0_7_mean": ("candidate_exposure_count_d0_7", "mean"),
        "candidate_exposure_count_d0_7_sum": ("candidate_exposure_count_d0_7", "sum"),
        "candidate_exposed_user_count_d0_7": ("candidate_exposure_count_d0_7", nonzero_count),
        "attendance_after_received_vote_rate": ("attendance_after_received_vote_yn_d0_28", rate),
        "point_use_after_received_vote_rate": ("point_use_after_received_vote_yn_d0_28", rate),
        "payment_after_received_vote_rate": ("payment_after_received_vote_yn_d0_28", rate),
        "sender_axis_score_mean": ("sender_axis_score", "mean"),
        "receiver_axis_score_mean": ("receiver_axis_score", "mean"),
        "network_axis_score_mean": ("network_axis_score", "mean"),
    }


def summarize_outcomes(df, group_cols):
    summary = (
        df.groupby(group_cols, as_index=False, observed=True)
        .agg(**outcome_aggregation_dict())
        .round(3)
    )
    summary = add_adjusted_mean_columns(summary).round(3)
    count_cols = [
        c
        for c in summary.columns
        if c.endswith("_count") or c.endswith("_sum") or c in ["user_count"]
    ]
    for col in count_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0).round(0).astype("int64")
    return summary


def build_segment_cluster_summaries(cluster_base, output_prefix=""):
    segment_specs = [
        (
            "sender_intensity_bucket_d0_7",
            "발신 행동축 강도 구간",
            "three_axis_sender_intensity_bucket_cluster_outcome",
        ),
        (
            "receiver_intensity_bucket_d0_7",
            "수신 반응축 강도 구간",
            "three_axis_receiver_intensity_bucket_cluster_outcome",
        ),
        (
            "candidate_exposure_bucket_d0_7",
            "관계망 환경축 후보 노출 강도 구간",
            "three_axis_candidate_exposure_bucket_cluster_outcome",
        ),
        (
            "friend_count_bucket",
            "친구 수 구간",
            "three_axis_friend_count_bucket_cluster_outcome",
        ),
    ]
    summaries = {}
    for segment_col, segment_label, file_name in segment_specs:
        if segment_col not in cluster_base.columns:
            continue
        segment_base = cluster_base.loc[cluster_base[segment_col].notna()].copy()
        if segment_base.empty:
            continue
        segment_total = (
            segment_base.groupby(segment_col, observed=False)["user_id"]
            .nunique()
            .rename("segment_user_count")
            .reset_index()
        )
        summary = summarize_outcomes(segment_base, [segment_col, "cluster_label"])
        summary = summary.loc[pd.to_numeric(summary["user_count"], errors="coerce").fillna(0) > 0].copy()
        summary = summary.merge(segment_total, on=segment_col, how="left")
        summary.insert(0, "segment_axis", segment_label)
        summary["cluster_share_within_segment_pct"] = (
            summary["user_count"] / summary["segment_user_count"].replace(0, np.nan) * 100
        ).fillna(0).round(2)
        summary = summary.sort_values([segment_col, "cluster_label"])
        save_table(summary, f"{output_prefix}{file_name}")
        summaries[segment_col] = summary
    return summaries


def apply_cluster_solution(cluster_base, scaled_df, labels):
    solution = cluster_base.copy()
    solution["cluster"] = np.asarray(labels).astype(int)
    solution = add_axis_scores(solution, scaled_df)
    label_map, rule_map, axis_profile = build_cluster_labels(solution)
    solution["cluster_label"] = solution["cluster"].map(label_map)

    cluster_order = (
        solution.groupby("cluster_label", as_index=False)
        .agg(
            user_count=("user_id", "nunique"),
            network_axis_score=("network_axis_score", "mean"),
            sender_axis_score=("sender_axis_score", "mean"),
            receiver_axis_score=("receiver_axis_score", "mean"),
        )
        .sort_values(["network_axis_score", "sender_axis_score", "receiver_axis_score"], ascending=[True, True, True])
        ["cluster_label"]
        .tolist()
    )
    solution["cluster_label"] = pd.Categorical(solution["cluster_label"], categories=cluster_order, ordered=True)
    solution = add_segment_buckets(solution)

    mapping = axis_profile.copy()
    mapping["cluster_label"] = mapping["cluster"].map(label_map)
    mapping["label_rule"] = mapping["cluster"].map(rule_map)
    mapping = mapping.sort_values("cluster_label")
    return solution, mapping, cluster_order


def save_cluster_solution_tables(cluster_base, scaled_df, available_features, mapping, output_prefix=""):
    save_table(mapping, f"{output_prefix}three_axis_cluster_label_mapping")

    scaled_with_cluster = scaled_df.copy()
    scaled_with_cluster["cluster_label"] = cluster_base["cluster_label"].values
    scaled_profile = (
        scaled_with_cluster.groupby("cluster_label", as_index=False, observed=False)
        .mean(numeric_only=True)
        .round(3)
    )
    save_table(scaled_profile, f"{output_prefix}three_axis_cluster_profile_standardized")

    raw_profile_cols = available_features + [c for c in PROFILE_ONLY_FEATURES if c in cluster_base.columns]
    raw_profile = (
        cluster_base.groupby("cluster_label", as_index=False, observed=False)
        .agg(user_count=("user_id", "nunique"), **{col: (col, "mean") for col in raw_profile_cols})
        .round(3)
    )
    save_table(raw_profile, f"{output_prefix}three_axis_cluster_profile_raw")

    outcome = summarize_outcomes(cluster_base, ["cluster_label"]).sort_values("cluster_label")
    outcome["cluster_share_pct"] = (outcome["user_count"] / outcome["user_count"].sum() * 100).round(2)
    save_table(outcome, f"{output_prefix}three_axis_cluster_outcome_summary")
    return raw_profile, outcome


# %% [markdown]
# ## 셀3-8. 3축 클러스터링 실행 함수


# %%
def run_three_axis_clustering(analysis):
    cell_note(
        "셀 12. 3축 표준화 클러스터링",
        "3축 입력 변수만 p99 보정, log1p, 표준화한 뒤 k 후보를 비교하고 최종 군집을 생성합니다.",
        "three_axis_cluster_*.csv, three_axis_cluster_*.png, 14_three_axis_user_clusters.csv",
    )
    cluster_base = analysis.loc[analysis["home_entry_population"]].copy()
    if cluster_base.empty:
        raise ValueError("홈화면 진입 모집단이 0명입니다. friend_count >= 4 and same_school_joined_user_count >= 40 조건을 확인하세요.")

    configured_features = SENDER_FEATURES + RECEIVER_FEATURES + NETWORK_FEATURES
    available_features = [
        col
        for col in configured_features
        if col in cluster_base.columns and pd.to_numeric(cluster_base[col], errors="coerce").fillna(0).std(ddof=0) > 0
    ]
    if len(available_features) < 4:
        raise ValueError(f"사용 가능한 클러스터링 변수가 너무 적습니다: {available_features}")

    feature_definition = build_feature_definition(available_features)
    save_table(feature_definition, "three_axis_cluster_feature_definition")

    raw_X, transformed_X, scaled_df, transform_summary = transform_cluster_features(cluster_base, available_features)
    save_table(transform_summary, "three_axis_cluster_feature_transform_summary")

    base_for_clustering = cluster_base.copy()
    best_k, k_metrics, labels = choose_cluster_k(scaled_df.values)
    if labels is None:
        final_model = build_model(best_k, len(cluster_base))
        labels = final_model.fit_predict(scaled_df.values)
    cluster_base, mapping, cluster_order = apply_cluster_solution(base_for_clustering, scaled_df, labels)

    k_metrics["selected_k"] = k_metrics["k"].eq(best_k)
    save_table(k_metrics, "three_axis_cluster_k_selection_metrics")

    raw_profile, outcome = save_cluster_solution_tables(cluster_base, scaled_df, available_features, mapping)

    cell_note(
        "셀 12-1. 축별 강도 구간과 친구 수 구간별 클러스터 결과",
        "발신 강도, 수신 강도, 후보 노출 강도, 친구 수 구간 안에서 클러스터 구성과 결과 지표를 비교합니다.",
        "three_axis_*_bucket_cluster_outcome.csv, three_axis_segment_cluster_heatmap.png",
    )
    segment_summaries = build_segment_cluster_summaries(cluster_base)

    export_cols = (
        ["user_id", "cluster", "cluster_label"]
        + available_features
        + [c for c in PROFILE_ONLY_FEATURES if c in cluster_base.columns]
        + [
            "sender_intensity_bucket_d0_7",
            "receiver_intensity_bucket_d0_7",
            "candidate_exposure_bucket_d0_7",
            "friend_count_bucket",
        ]
        + ["sender_axis_score", "receiver_axis_score", "network_axis_score"]
        + [c for c in OUTCOME_COLUMNS if c in cluster_base.columns]
    )
    save_processed(cluster_base[export_cols], "14_three_axis_user_clusters.csv")

    make_cluster_figures(cluster_base, scaled_df, k_metrics, best_k, available_features, cluster_order, segment_summaries)
    build_cluster_interpretation(cluster_base, outcome, raw_profile, mapping, segment_summaries, k_metrics)

    for compare_k in COMPARE_THREE_AXIS_KS:
        if compare_k == best_k or compare_k < 2 or compare_k > MAX_THREE_AXIS_CLUSTER_K:
            continue
        cell_note(
            f"셀 12-3. k={compare_k} 비교 클러스터링",
            f"기본 선택 k={best_k} 결과는 유지하고, k={compare_k}를 민감도 분석용으로 별도 생성합니다.",
            f"k{compare_k}_three_axis_cluster_*.csv, k{compare_k}_three_axis_cluster_*.png, k{compare_k}_14_three_axis_user_clusters.csv",
        )
        compare_model = build_model(compare_k, len(base_for_clustering))
        compare_labels = compare_model.fit_predict(scaled_df.values)
        compare_base, compare_mapping, compare_order = apply_cluster_solution(base_for_clustering, scaled_df, compare_labels)
        compare_prefix = f"k{compare_k}_"
        save_cluster_solution_tables(
            compare_base,
            scaled_df,
            available_features,
            compare_mapping,
            output_prefix=compare_prefix,
        )
        compare_segment_summaries = build_segment_cluster_summaries(compare_base, output_prefix=compare_prefix)
        save_processed(compare_base[export_cols], f"{compare_prefix}14_three_axis_user_clusters.csv")
        make_cluster_figures(
            compare_base,
            scaled_df,
            k_metrics,
            compare_k,
            available_features,
            compare_order,
            compare_segment_summaries,
            output_prefix=compare_prefix,
            include_k_selection=False,
            title_suffix=f" (k={compare_k} 비교)",
        )

    if FIGURE_CAPTION_ROWS:
        save_table(pd.DataFrame(FIGURE_CAPTION_ROWS), "three_axis_figure_interpretations")
    if GRAPH_VALIDATION_ROWS:
        save_table(pd.DataFrame(GRAPH_VALIDATION_ROWS), "three_axis_graph_data_validation")
    return cluster_base, outcome, k_metrics


# %% [markdown]
# ## 셀3-9. 클러스터링 결과 시각화 함수


# %%
def make_cluster_figures(
    cluster_base,
    scaled_df,
    k_metrics,
    best_k,
    available_features,
    cluster_order,
    segment_summaries,
    output_prefix="",
    include_k_selection=True,
    title_suffix="",
):
    def figure_name(name):
        return f"{output_prefix}{name}"

    cluster_outcome = summarize_outcomes(cluster_base, ["cluster_label"]).sort_values("cluster_label")
    palette = dict(zip(cluster_order, sns.color_palette("Set2", n_colors=max(len(cluster_order), 1))))

    if include_k_selection:
        save_k_selection_figure(k_metrics, selected_k=best_k)

    pca_idx = sampled_indices(len(cluster_base), PCA_PLOT_SAMPLE_SIZE, RANDOM_STATE)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(scaled_df.values)
    pca_df = pd.DataFrame(
        {
            "PC1": pcs[pca_idx, 0],
            "PC2": pcs[pca_idx, 1],
            "cluster_label": cluster_base["cluster_label"].astype("string").iloc[pca_idx].values,
        }
    )
    validate_graph_data(
        figure_name("three_axis_cluster_pca_2d"),
        pca_df,
        required_cols=["PC1", "PC2", "cluster_label"],
        value_cols=["PC1", "PC2"],
        allow_all_zero=False,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster_label",
        hue_order=cluster_order,
        palette=palette,
        alpha=0.65,
        s=18,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(f"3축 클러스터 PCA{title_suffix}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(title="군집", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    save_fig(fig, figure_name("three_axis_cluster_pca_2d"))

    heat = scaled_df.copy()
    heat["cluster_label"] = cluster_base["cluster_label"].values
    heat_profile = heat.groupby("cluster_label", observed=False).mean()
    heat_profile = heat_profile.rename(columns={col: label_for(col) for col in heat_profile.columns})
    validate_graph_data(
        figure_name("three_axis_cluster_profile_heatmap"),
        heat_profile.reset_index(),
        required_cols=["cluster_label"],
        value_cols=[c for c in heat_profile.columns],
        allow_all_zero=False,
    )
    fig, ax = plt.subplots(figsize=(max(12, len(available_features) * 0.7), max(4, len(cluster_order) * 0.7)))
    sns.heatmap(heat_profile, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.4, ax=ax)
    ax.set_title(f"군집별 표준화 입력 변수 평균{title_suffix}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, figure_name("three_axis_cluster_profile_heatmap"))

    axis_score_cols = [
        "sender_axis_score_mean",
        "receiver_axis_score_mean",
        "network_axis_score_mean",
    ]
    axis_score_plot = cluster_outcome.melt(
        id_vars=["cluster_label"],
        value_vars=[c for c in axis_score_cols if c in cluster_outcome.columns],
        var_name="axis",
        value_name="mean_axis_score",
    )
    axis_score_plot["axis_label"] = axis_score_plot["axis"].map(
        {
            "sender_axis_score_mean": "발신 행동축",
            "receiver_axis_score_mean": "수신 반응축",
            "network_axis_score_mean": "관계망 환경축",
        }
    )
    validate_graph_data(
        figure_name("three_axis_cluster_axis_scores"),
        axis_score_plot,
        required_cols=["cluster_label", "axis_label"],
        value_cols=["mean_axis_score"],
        allow_all_zero=False,
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=axis_score_plot,
        x="cluster_label",
        y="mean_axis_score",
        hue="axis_label",
        palette=["#3B8F63", "#2E5EAA", "#9A6A3A"],
        ax=ax,
    )
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title(f"클러스터별 3축 평균 점수{title_suffix}")
    ax.set_xlabel("클러스터")
    ax.set_ylabel("평균 표준화 점수")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="축", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    save_fig(fig, figure_name("three_axis_cluster_axis_scores"))

    outcome_plot = (
        cluster_base.groupby("cluster_label", as_index=False, observed=False)
        .agg(
            retention_d1_28=("retention_d1_28", rate),
            payment_d0_28=("paid_flag_d0_28", rate),
            attendance_after_received=("attendance_after_received_vote_yn_d0_28", rate),
        )
        .melt(id_vars="cluster_label", var_name="metric", value_name="rate_pct")
    )
    outcome_plot["metric_label"] = outcome_plot["metric"].map(
        {
            "retention_d1_28": "D1-D28 재방문율",
            "payment_d0_28": "D0-D28 결제율",
            "attendance_after_received": "수신 이후 출석률",
        }
    )
    validation_status = validate_graph_data(
        figure_name("three_axis_cluster_outcome_rates"),
        outcome_plot,
        required_cols=["cluster_label", "metric_label"],
        value_cols=["rate_pct"],
        allow_all_zero=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=outcome_plot,
        x="metric_label",
        y="rate_pct",
        hue="cluster_label",
        hue_order=cluster_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title(f"군집별 결과 지표 검증{title_suffix}")
    ax.set_xlabel("")
    ax.set_ylabel("%")
    ax.legend(title="군집", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    annotate_all_zero_if_needed(ax, validation_status)
    save_fig(fig, figure_name("three_axis_cluster_outcome_rates"))

    outcome_heatmap_cols = [
        "retention_d1_28_rate",
        "retention_d8_28_rate",
        "payment_d0_28_rate",
        "payment_d8_28_rate",
        "payment_d0_28_count_paid_user_mean",
        "payment_d0_28_amount_paid_user_mean",
        "active_days_d0_28_mean",
        "active_days_d8_28_mean",
    ]
    outcome_heatmap = cluster_outcome.set_index("cluster_label")[
        [c for c in outcome_heatmap_cols if c in cluster_outcome.columns]
    ].copy()
    outcome_heatmap = outcome_heatmap.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    outcome_heatmap = outcome_heatmap.rename(
        columns={
            "retention_d1_28_rate": "재방문율\nD1-D28",
            "retention_d8_28_rate": "재방문율\nD8-D28",
            "payment_d0_28_rate": "결제율\nD0-D28",
            "payment_d8_28_rate": "결제율\nD8-D28",
            "payment_d0_28_count_paid_user_mean": "결제자 평균\n결제 빈도\nD0-D28",
            "payment_d0_28_amount_paid_user_mean": "결제자 평균\n구매 하트\nD0-D28",
            "active_days_d0_28_mean": "평균 활동일수\nD0-D28",
            "active_days_d8_28_mean": "평균 활동일수\nD8-D28",
        }
    )
    validation_status = validate_graph_data(
        figure_name("three_axis_cluster_outcome_heatmap"),
        outcome_heatmap.reset_index(),
        required_cols=["cluster_label"],
        value_cols=[c for c in outcome_heatmap.columns],
        allow_all_zero=True,
    )
    fig, ax = plt.subplots(
        figsize=(max(12, len(outcome_heatmap.columns) * 1.45), max(4.8, len(outcome_heatmap) * 0.9 + 1.8))
    )
    sns.heatmap(
        outcome_heatmap,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        annot_kws={"fontsize": 9},
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title(f"클러스터별 핵심 결과 지표 히트맵{title_suffix}", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=10)
    ax.tick_params(axis="y", pad=8)
    annotate_all_zero_if_needed(ax, validation_status)
    save_fig(fig, figure_name("three_axis_cluster_outcome_heatmap"))

    payment_count_cols = [
        "payment_d0_28_count_mean",
        "payment_d0_28_count_paid_user_mean",
        "payment_d8_28_count_mean",
        "payment_d8_28_count_paid_user_mean",
    ]
    payment_count_plot = cluster_outcome.melt(
        id_vars=["cluster_label"],
        value_vars=[c for c in payment_count_cols if c in cluster_outcome.columns],
        var_name="metric",
        value_name="mean_count",
    )
    payment_count_plot["metric_label"] = payment_count_plot["metric"].map(
        {
            "payment_d0_28_count_mean": "전체 평균 결제 빈도 D0-D28",
            "payment_d0_28_count_paid_user_mean": "결제자 평균 결제 빈도 D0-D28",
            "payment_d8_28_count_mean": "전체 평균 결제 빈도 D8-D28",
            "payment_d8_28_count_paid_user_mean": "결제자 평균 결제 빈도 D8-D28",
        }
    )
    validation_status = validate_graph_data(
        figure_name("three_axis_cluster_payment_count"),
        payment_count_plot,
        required_cols=["cluster_label", "metric_label"],
        value_cols=["mean_count"],
        allow_all_zero=True,
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=payment_count_plot,
        x="metric_label",
        y="mean_count",
        hue="cluster_label",
        hue_order=cluster_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title(f"클러스터별 평균 결제 빈도{title_suffix}")
    ax.set_xlabel("")
    ax.set_ylabel("평균 결제 횟수")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="군집", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    annotate_all_zero_if_needed(ax, validation_status)
    save_fig(fig, figure_name("three_axis_cluster_payment_count"))

    payment_amount_cols = [
        "payment_d0_28_amount_mean",
        "payment_d0_28_amount_paid_user_mean",
        "payment_d8_28_amount_mean",
        "payment_d8_28_amount_paid_user_mean",
    ]
    payment_amount_plot = cluster_outcome.melt(
        id_vars=["cluster_label"],
        value_vars=[c for c in payment_amount_cols if c in cluster_outcome.columns],
        var_name="metric",
        value_name="mean_amount",
    )
    payment_amount_plot["metric_label"] = payment_amount_plot["metric"].map(
        {
            "payment_d0_28_amount_mean": "전체 평균 추정 구매 하트 수 D0-D28",
            "payment_d0_28_amount_paid_user_mean": "결제자 평균 추정 구매 하트 수 D0-D28",
            "payment_d8_28_amount_mean": "전체 평균 추정 구매 하트 수 D8-D28",
            "payment_d8_28_amount_paid_user_mean": "결제자 평균 추정 구매 하트 수 D8-D28",
        }
    )
    validation_status = validate_graph_data(
        figure_name("three_axis_cluster_payment_amount"),
        payment_amount_plot,
        required_cols=["cluster_label", "metric_label"],
        value_cols=["mean_amount"],
        allow_all_zero=True,
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=payment_amount_plot,
        x="metric_label",
        y="mean_amount",
        hue="cluster_label",
        hue_order=cluster_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title(f"클러스터별 평균 추정 구매 하트 수{title_suffix}")
    ax.set_xlabel("")
    ax.set_ylabel("평균 추정 구매 하트 수")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="군집", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    annotate_all_zero_if_needed(ax, validation_status)
    save_fig(fig, figure_name("three_axis_cluster_payment_amount"))

    active_cols = [
        "active_days_d0_28_mean",
        "active_days_d0_28_active_user_mean",
        "active_days_d8_28_mean",
        "active_days_d8_28_active_user_mean",
    ]
    active_plot = cluster_outcome.melt(
        id_vars=["cluster_label"],
        value_vars=[c for c in active_cols if c in cluster_outcome.columns],
        var_name="metric",
        value_name="mean_days",
    )
    active_plot["metric_label"] = active_plot["metric"].map(
        {
            "active_days_d0_28_mean": "전체 평균 활동일수 D0-D28",
            "active_days_d0_28_active_user_mean": "활동 유저 평균 활동일수 D0-D28",
            "active_days_d8_28_mean": "전체 평균 활동일수 D8-D28",
            "active_days_d8_28_active_user_mean": "활동 유저 평균 활동일수 D8-D28",
        }
    )
    validation_status = validate_graph_data(
        figure_name("three_axis_cluster_active_days"),
        active_plot,
        required_cols=["cluster_label", "metric_label"],
        value_cols=["mean_days"],
        allow_all_zero=True,
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=active_plot,
        x="metric_label",
        y="mean_days",
        hue="cluster_label",
        hue_order=cluster_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title(f"클러스터별 평균 누적 활동일수{title_suffix}")
    ax.set_xlabel("")
    ax.set_ylabel("평균 활동일수")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="군집", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    annotate_all_zero_if_needed(ax, validation_status)
    save_fig(fig, figure_name("three_axis_cluster_active_days"))

    heat_rows = []
    heat_source = {
        "발신 강도": "sender_intensity_bucket_d0_7",
        "수신 강도": "receiver_intensity_bucket_d0_7",
        "후보 노출 강도": "candidate_exposure_bucket_d0_7",
        "친구 수": "friend_count_bucket",
    }
    for section, segment_col in heat_source.items():
        summary = segment_summaries.get(segment_col)
        if summary is None or summary.empty:
            continue
        for _, row in summary.iterrows():
            heat_rows.append(
                {
                    "segment": f"{section}: {row[segment_col]}",
                    "cluster_label": row["cluster_label"],
                    "cluster_share_within_segment_pct": row["cluster_share_within_segment_pct"],
                }
            )
    if heat_rows:
        heat_df = pd.DataFrame(heat_rows)
        validate_graph_data(
            figure_name("three_axis_segment_cluster_heatmap"),
            heat_df,
            required_cols=["segment", "cluster_label"],
            value_cols=["cluster_share_within_segment_pct"],
            allow_all_zero=False,
        )
        heat_pivot = heat_df.pivot_table(
            index="segment",
            columns="cluster_label",
            values="cluster_share_within_segment_pct",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        heat_pivot = heat_pivot[[c for c in cluster_order if c in heat_pivot.columns]]
        fig, ax = plt.subplots(figsize=(max(9, len(cluster_order) * 1.4), max(5, len(heat_pivot) * 0.45)))
        sns.heatmap(heat_pivot, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=0.4, ax=ax)
        ax.set_title(f"강도/친구 수 구간별 클러스터 구성비{title_suffix}")
        ax.set_xlabel("클러스터")
        ax.set_ylabel("구간")
        save_fig(fig, figure_name("three_axis_segment_cluster_heatmap"))
    else:
        validate_graph_data(
            figure_name("three_axis_segment_cluster_heatmap"),
            pd.DataFrame(columns=["segment", "cluster_label", "cluster_share_within_segment_pct"]),
            required_cols=["segment", "cluster_label"],
            value_cols=["cluster_share_within_segment_pct"],
            allow_all_zero=False,
        )

# %% [markdown]
# ## 셀3-10. 클러스터링 결과 해석 함수


# %%
def build_cluster_interpretation(cluster_base, outcome, raw_profile, mapping, segment_summaries, k_metrics):
    global FINAL_INTERPRETATION_MARKDOWN, FINAL_INTERPRETATION_ROWS

    cell_note(
        "셀 12-2. 클러스터링 결과 자동 해석",
        "클러스터별 3축 특징과 재방문, 결제, 활동일수 결과를 문장형 해석으로 정리합니다.",
        "three_axis_cluster_interpretation_summary.csv, three_axis_segment_cluster_interpretation.csv, three_axis_final_interpretation.md",
    )

    def numeric_value(row, col, default=np.nan):
        if col not in row.index:
            return default
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        return default if pd.isna(value) else float(value)

    def fmt_pct(value):
        return "NA" if pd.isna(value) else f"{value:.2f}%"

    def fmt_num(value, digits=2):
        return "NA" if pd.isna(value) else f"{value:.{digits}f}"

    def top_cluster(df, metric_col):
        if metric_col not in df.columns or df.empty:
            return None
        metric = pd.to_numeric(df[metric_col], errors="coerce")
        if metric.notna().sum() == 0:
            return None
        row = df.loc[metric.idxmax()]
        return str(row["cluster_label"]), float(metric.loc[metric.idxmax()])

    raw_merge_cols = [c for c in raw_profile.columns if c != "user_count"]
    interpret_base = outcome.merge(raw_profile[raw_merge_cols], on="cluster_label", how="left")
    mapping_cols = [c for c in ["cluster_label", "label_rule"] if c in mapping.columns]
    if mapping_cols:
        interpret_base = interpret_base.merge(mapping[mapping_cols], on="cluster_label", how="left")

    total_users = max(float(pd.to_numeric(outcome["user_count"], errors="coerce").fillna(0).sum()), 1.0)
    rows = []
    for _, row in interpret_base.iterrows():
        axis_scores = {
            "발신 행동축": numeric_value(row, "sender_axis_score_mean"),
            "수신 반응축": numeric_value(row, "receiver_axis_score_mean"),
            "관계망 환경축": numeric_value(row, "network_axis_score_mean"),
        }
        valid_axis_scores = {k: v for k, v in axis_scores.items() if not pd.isna(v)}
        dominant_axis = max(valid_axis_scores, key=valid_axis_scores.get) if valid_axis_scores else "해석 보류"
        user_count = numeric_value(row, "user_count", 0)
        cluster_share = numeric_value(row, "cluster_share_pct", user_count / total_users * 100)
        retention = numeric_value(row, "retention_d1_28_rate")
        payment = numeric_value(row, "payment_d0_28_rate")
        active_days = numeric_value(row, "active_days_d0_28_mean")
        payment_count = numeric_value(row, "payment_d0_28_count_paid_user_mean")
        payment_amount = numeric_value(row, "payment_d0_28_amount_paid_user_mean")
        sent_vote = numeric_value(row, "sent_vote_count_d0_7")
        received_vote = numeric_value(row, "received_vote_count_d0_7")
        friend_count = numeric_value(row, "friend_count")
        candidate_exposure = numeric_value(row, "candidate_exposure_count_d0_7")
        label_rule = str(row.get("label_rule", "")).strip()

        interpretation = (
            f"{row['cluster_label']}: 전체 {fmt_pct(cluster_share)}({int(user_count):,}명) 규모이며, "
            f"가장 두드러진 축은 {dominant_axis}입니다. "
            f"D1-D28 재방문율은 {fmt_pct(retention)}, D0-D28 결제율은 {fmt_pct(payment)}, "
            f"D0-D28 평균 활동일수는 {fmt_num(active_days)}일입니다. "
            f"결제자 기준 평균 결제 빈도는 {fmt_num(payment_count)}회, "
            f"추정 구매 하트 수는 {fmt_num(payment_amount)}개입니다."
        )
        if label_rule:
            interpretation += f" 군집명 근거는 '{label_rule}'입니다."

        rows.append(
            {
                "cluster_label": row["cluster_label"],
                "user_count": int(user_count),
                "cluster_share_pct": round(cluster_share, 2),
                "dominant_axis": dominant_axis,
                "sender_axis_score_mean": round(axis_scores["발신 행동축"], 3),
                "receiver_axis_score_mean": round(axis_scores["수신 반응축"], 3),
                "network_axis_score_mean": round(axis_scores["관계망 환경축"], 3),
                "sent_vote_count_d0_7_mean": round(sent_vote, 3),
                "received_vote_count_d0_7_mean": round(received_vote, 3),
                "friend_count_mean": round(friend_count, 3),
                "candidate_exposure_count_d0_7_mean": round(candidate_exposure, 3),
                "retention_d1_28_rate": round(retention, 3),
                "payment_d0_28_rate": round(payment, 3),
                "payment_d0_28_count_paid_user_mean": round(payment_count, 3),
                "payment_d0_28_amount_paid_user_mean": round(payment_amount, 3),
                "active_days_d0_28_mean": round(active_days, 3),
                "interpretation": interpretation,
            }
        )

    interpretation_df = pd.DataFrame(rows)
    FINAL_INTERPRETATION_ROWS = rows
    save_table(interpretation_df, "three_axis_cluster_interpretation_summary")

    segment_rows = []
    segment_name_map = {
        "sender_intensity_bucket_d0_7": "발신 행동축 강도 구간",
        "receiver_intensity_bucket_d0_7": "수신 반응축 강도 구간",
        "candidate_exposure_bucket_d0_7": "관계망 환경축 후보 노출 강도 구간",
        "friend_count_bucket": "친구 수 구간",
    }
    for segment_col, summary in segment_summaries.items():
        if summary is None or summary.empty or segment_col not in summary.columns:
            continue
        for segment_value, segment_df in summary.groupby(segment_col, observed=False):
            if segment_df.empty or "cluster_share_within_segment_pct" not in segment_df.columns:
                continue
            share = pd.to_numeric(segment_df["cluster_share_within_segment_pct"], errors="coerce")
            if share.notna().sum() == 0:
                continue
            row = segment_df.loc[share.idxmax()]
            segment_axis = segment_name_map.get(segment_col, str(row.get("segment_axis", segment_col)))
            interpretation = (
                f"{segment_axis} '{segment_value}' 구간에서는 {row['cluster_label']} 비중이 "
                f"{fmt_pct(numeric_value(row, 'cluster_share_within_segment_pct'))}로 가장 큽니다. "
                f"이 구간의 해당 군집 D1-D28 재방문율은 {fmt_pct(numeric_value(row, 'retention_d1_28_rate'))}, "
                f"D0-D28 결제율은 {fmt_pct(numeric_value(row, 'payment_d0_28_rate'))}, "
                f"D0-D28 평균 활동일수는 {fmt_num(numeric_value(row, 'active_days_d0_28_mean'))}일입니다."
            )
            segment_rows.append(
                {
                    "segment_axis": segment_axis,
                    "segment_column": segment_col,
                    "segment_value": segment_value,
                    "top_cluster_label": row["cluster_label"],
                    "top_cluster_share_within_segment_pct": round(
                        numeric_value(row, "cluster_share_within_segment_pct"), 2
                    ),
                    "top_cluster_user_count": int(numeric_value(row, "user_count", 0)),
                    "retention_d1_28_rate": round(numeric_value(row, "retention_d1_28_rate"), 3),
                    "payment_d0_28_rate": round(numeric_value(row, "payment_d0_28_rate"), 3),
                    "active_days_d0_28_mean": round(numeric_value(row, "active_days_d0_28_mean"), 3),
                    "interpretation": interpretation,
                }
            )
    segment_interpretation_df = pd.DataFrame(segment_rows)
    save_table(segment_interpretation_df, "three_axis_segment_cluster_interpretation")

    selected = k_metrics.loc[k_metrics["selected_k"]].iloc[0] if "selected_k" in k_metrics.columns else k_metrics.iloc[0]
    largest = top_cluster(outcome, "user_count")
    best_retention = top_cluster(outcome, "retention_d1_28_rate")
    best_payment = top_cluster(outcome, "payment_d0_28_rate")
    best_active = top_cluster(outcome, "active_days_d0_28_mean")
    best_amount = top_cluster(outcome, "payment_d0_28_amount_paid_user_mean")

    lines = [
        "### 선택 k와 핵심 군집",
        "",
        f"- 선택된 k는 {int(selected['k'])}이며, 표본 실루엣 점수는 {float(selected['silhouette_score']):.4f}, 최소 군집 비율은 {float(selected['min_cluster_share_pct']):.2f}%입니다.",
    ]
    if largest:
        lines.append(f"- 가장 큰 군집은 {largest[0]}이며 전체 {largest[1]:,.0f}명입니다.")
    if best_retention:
        lines.append(f"- D1-D28 재방문율이 가장 높은 군집은 {best_retention[0]}({best_retention[1]:.2f}%)입니다.")
    if best_payment:
        lines.append(f"- D0-D28 결제율이 가장 높은 군집은 {best_payment[0]}({best_payment[1]:.2f}%)입니다.")
    if best_active:
        lines.append(f"- D0-D28 평균 활동일수가 가장 높은 군집은 {best_active[0]}({best_active[1]:.2f}일)입니다.")
    if best_amount:
        lines.append(f"- 결제자 기준 추정 구매 하트 수가 가장 높은 군집은 {best_amount[0]}({best_amount[1]:.2f}개)입니다.")

    lines.extend(["", "### 군집별 해석", ""])
    for row in rows:
        lines.append(f"- {row['interpretation']}")

    if segment_rows:
        lines.extend(["", "### 구간별 우세 군집 해석", ""])
        for row in segment_rows:
            lines.append(f"- {row['interpretation']}")

    FINAL_INTERPRETATION_MARKDOWN = "\n".join(lines)
    markdown_path = TABLE_DIR / "three_axis_final_interpretation.md"
    markdown_path.write_text("# 3축 클러스터링 자동 해석\n\n" + FINAL_INTERPRETATION_MARKDOWN + "\n", encoding="utf-8")
    print(f"saved interpretation: {markdown_path}")
    return FINAL_INTERPRETATION_MARKDOWN


# %% [markdown]
# ## 셀3-11. 전체 분석 실행


# %%
def main():
    global LAST_CLUSTER_BASE, LAST_OUTCOME, LAST_K_METRICS

    setup_colab_environment()
    init_output_dirs()
    setup_plot_style()
    print("[셀3-11] 데이터 경로를 확인합니다.")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"OUT_DIR = {OUT_DIR}")
    if not (DATA_DIR / "accounts_user_master.csv").exists():
        print(f"DATA_DIR exists = {DATA_DIR.exists()}")
        if DATA_DIR.exists():
            csv_preview = sorted(path.name for path in DATA_DIR.glob("*.csv"))[:20]
            print(f"DATA_DIR csv preview = {csv_preview}")
        raise FileNotFoundError(f"DATA_DIR에서 accounts_user_master.csv를 찾지 못했습니다: {DATA_DIR}")
    cell_note(
        "셀 00. 관찰 기간과 결과 지표 정의",
        "클러스터링 입력은 D0-D7 초기 3축 변수로 제한하고, 재방문/결제/활동일수는 D1-D28, D0-D28, D8-D28 사후 결과로 분리합니다. 원천 파일과 필수 컬럼 참조도 먼저 검증합니다.",
        "three_axis_observation_window_definition.csv, three_axis_source_reference_check.csv",
    )
    save_observation_window_definition()
    check_source_references()

    users = build_users_population()
    attendance_metrics, attendance = build_attendance_metrics(users)
    payment_metrics, payment = build_payment_metrics(users)
    sender_metrics, receiver_metrics, question_sent_activity, received_read_activity, first_received = build_vote_metrics(users)
    friend_metrics, friend_activity = build_friend_request_metrics(users)
    point_metrics, point_activity = build_point_metrics(users)
    active_day_metrics, network_metrics, activity_days = build_active_day_and_network_metrics(
        users,
        [question_sent_activity, received_read_activity, friend_activity, point_activity],
    )
    hackle_metrics = build_hackle_metrics(users)
    candidate_metrics = build_candidate_exposure_metrics(users, receiver_metrics)
    after_received_metrics = build_after_received_metrics(first_received, attendance, point_activity, payment, users)

    analysis = prepare_analysis_table(
        users,
        attendance_metrics,
        payment_metrics,
        sender_metrics,
        receiver_metrics,
        friend_metrics,
        point_metrics,
        active_day_metrics,
        network_metrics,
        hackle_metrics,
        candidate_metrics,
        after_received_metrics,
    )
    cluster_base, outcome, k_metrics = run_three_axis_clustering(analysis)
    LAST_CLUSTER_BASE = cluster_base
    LAST_OUTCOME = outcome
    LAST_K_METRICS = k_metrics

    display_analysis_results(outcome, k_metrics)
    display_all_saved_figures()

    print("완료")
    print(f"OUT_DIR = {OUT_DIR}")
    print(f"clustered users = {cluster_base['user_id'].nunique():,}")
    if STOP_BEFORE_GITHUB_STEP_FOR_REVIEW:
        print("검토 모드: GitHub 공유/commit/push 전에 멈췄습니다.")
        print("그래프와 표를 확인한 뒤 publish_reviewed_outputs()를 실행하세요.")
        print("실제 push까지 하려면 먼저 RUN_GITHUB_PUSH = True로 바꾸세요.")
        return cluster_base, outcome, k_metrics

    if "publish_reviewed_outputs" not in globals():
        print("GitHub 공유 함수는 셀3-12, 셀3-13 실행 후 사용할 수 있습니다.")
        return cluster_base, outcome, k_metrics
    publish_reviewed_outputs()
    return cluster_base, outcome, k_metrics


if __name__ == "__main__":
    print("[셀3-11] 전체 분석을 시작합니다.")
    cluster_base, outcome, k_metrics = main()


# %% [markdown]
# ## 셀3-12. GitHub 공유용 결과 묶기


# %%
def build_github_share(outcome, k_metrics):
    cell_note(
        "셀 12. GitHub 공유용 결과 묶기",
        "PNG, 결과표, 셀별 설명, 그래프 해석 README를 GitHub에 올리기 좋은 폴더와 zip 파일로 묶습니다.",
        "github_share 폴더, three_axis_clustering_github_share.zip",
    )
    if ANALYSIS_NOTES:
        save_table(pd.DataFrame(ANALYSIS_NOTES), "three_axis_analysis_cell_notes")
    if FIGURE_CAPTION_ROWS:
        save_table(pd.DataFrame(FIGURE_CAPTION_ROWS), "three_axis_figure_interpretations")
    if GRAPH_VALIDATION_ROWS:
        save_table(pd.DataFrame(GRAPH_VALIDATION_ROWS), "three_axis_graph_data_validation")
    if FINAL_INTERPRETATION_ROWS:
        save_table(pd.DataFrame(FINAL_INTERPRETATION_ROWS), "three_axis_cluster_interpretation_summary")
    if FINAL_INTERPRETATION_MARKDOWN:
        (TABLE_DIR / "three_axis_final_interpretation.md").write_text(
            "# 3축 클러스터링 자동 해석\n\n" + FINAL_INTERPRETATION_MARKDOWN + "\n",
            encoding="utf-8",
        )
    share_dir = OUT_DIR / "github_share"
    if share_dir.exists():
        shutil.rmtree(share_dir)
    share_dir.mkdir(parents=True, exist_ok=True)

    for source_dir in [FIG_DIR, TABLE_DIR]:
        target = share_dir / source_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for source_path in sorted(source_dir.glob("*")):
            if source_path.is_file():
                shutil.copy2(source_path, target / source_path.name)

    processed_manifest = pd.DataFrame(
        [
            {
                "file_name": path.name,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "github_share_included": False,
            }
            for path in sorted(PROCESSED_DIR.glob("*"))
            if path.is_file()
        ]
    )
    processed_manifest.to_csv(share_dir / "processed_files_manifest.csv", index=False, encoding="utf-8-sig")

    readme_path = share_dir / "README_three_axis_clustering.md"
    selected = k_metrics.loc[k_metrics["selected_k"]].iloc[0]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# 3축 통합 유저 클러스터링 결과\n\n")
        f.write("원본 raw CSV와 대용량 processed CSV는 GitHub 공유 묶음에 포함하지 않습니다.\n\n")
        f.write("## 경로\n")
        f.write(f"- Google Drive data path: `{DATA_DIR}`\n")
        f.write(f"- GitHub repo: `{GITHUB_REPO}`\n")
        f.write(f"- Branch: `{BRANCH_NAME}`\n\n")
        f.write("## 클러스터링 입력 축\n")
        f.write("- 발신 행동축: D0-D7 보낸 투표 수, 발신 활동일수, 선택한 유저 수, 질문 수, 완료율, Hackle 보완 지표\n")
        f.write("- 수신 반응축: D0-D7 받은 투표 수, 수신 발생일수, 열람 수/열람률, 미열람 수신, 초기 수신 빠름 점수\n")
        f.write("- 관계망 환경축: 친구 수, 같은 반/학년 활성 유저 수, 후보 노출, 후보 노출 대비 선택률, 미확인 Ping 수\n\n")
        f.write("## 전처리\n")
        f.write("- count형 변수: p99 winsorize 후 log1p 변환\n")
        f.write("- rate형 변수: 0-1 범위 clip\n")
        f.write("- 모든 입력 변수: StandardScaler 표준화\n")
        f.write("- 재방문/결제/활동일수 결과 지표는 클러스터링 입력에서 제외하고 사후 검증에만 사용\n\n")
        f.write("## 선택된 k\n")
        f.write(f"- selected k: `{int(selected['k'])}`\n")
        f.write(f"- selection reason: {selected['selection_reason']}\n")
        f.write(f"- sampled silhouette: `{selected['silhouette_score']:.4f}`\n")
        f.write(f"- min cluster share: `{selected['min_cluster_share_pct']:.2f}%`\n\n")
        f.write("## 핵심 산출물\n")
        f.write("- `tables/three_axis_cluster_feature_definition.csv`\n")
        f.write("- `tables/three_axis_source_reference_check.csv`\n")
        f.write("- `tables/three_axis_graph_data_validation.csv`\n")
        f.write("- `tables/three_axis_cluster_k_selection_metrics.csv`\n")
        f.write("- `tables/three_axis_cluster_label_mapping.csv`\n")
        f.write("- `tables/three_axis_cluster_profile_raw.csv`\n")
        f.write("- `tables/three_axis_cluster_profile_standardized.csv`\n")
        f.write("- `tables/three_axis_cluster_outcome_summary.csv`\n")
        f.write("- `tables/three_axis_cluster_interpretation_summary.csv`\n")
        f.write("- `tables/three_axis_segment_cluster_interpretation.csv`\n")
        f.write("- `tables/three_axis_final_interpretation.md`\n")
        f.write("- `tables/three_axis_sender_intensity_bucket_cluster_outcome.csv`\n")
        f.write("- `tables/three_axis_receiver_intensity_bucket_cluster_outcome.csv`\n")
        f.write("- `tables/three_axis_candidate_exposure_bucket_cluster_outcome.csv`\n")
        f.write("- `tables/three_axis_friend_count_bucket_cluster_outcome.csv`\n")
        f.write("- `figures/three_axis_cluster_pca_2d.png`\n")
        f.write("- `figures/three_axis_cluster_profile_heatmap.png`\n")
        f.write("- `figures/three_axis_cluster_axis_scores.png`\n")
        f.write("- `figures/three_axis_cluster_outcome_rates.png`\n")
        f.write("- `figures/three_axis_cluster_outcome_heatmap.png`\n")
        f.write("- `figures/three_axis_cluster_payment_count.png`\n")
        f.write("- `figures/three_axis_cluster_payment_amount.png`\n")
        f.write("- `figures/three_axis_cluster_active_days.png`\n")
        f.write("- `figures/three_axis_segment_cluster_heatmap.png`\n\n")
        f.write("## 셀별 설명\n\n")
        for note in ANALYSIS_NOTES:
            f.write(f"### {note['title']}\n")
            f.write(f"- 목적: {note['purpose']}\n")
            f.write(f"- 산출물: {note['output']}\n\n")
        f.write("## 그래프 해석\n\n")
        for row in FIGURE_CAPTION_ROWS:
            f.write(f"- `{row['figure_file']}`: {row['interpretation']}\n")
        f.write("\n")
        if FINAL_INTERPRETATION_MARKDOWN:
            f.write("## 자동 해석 요약\n\n")
            f.write(FINAL_INTERPRETATION_MARKDOWN)
            f.write("\n\n")
        f.write("## 군집별 결과 요약\n\n")
        f.write("```text\n")
        f.write(outcome.to_string(index=False))
        f.write("\n```")
        f.write("\n")

    zip_path = shutil.make_archive(str(OUT_DIR / "three_axis_clustering_github_share"), "zip", share_dir)
    print("GitHub share folder:", share_dir)
    print("GitHub share zip:", zip_path)

    if COPY_OUTPUTS_TO_REPO_IF_AVAILABLE and REPO_DIR.exists():
        target_dir = REPO_DIR / "analysis" / "three_axis_clustering"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(share_dir, target_dir / "github_share", dirs_exist_ok=True)
        print(f"Copied GitHub share to: {target_dir / 'github_share'}")

    return share_dir

# %% [markdown]
# ## 셀3-13. 선택 실행: GitHub commit/push


# %%
def maybe_push_to_github():
    if not RUN_GITHUB_PUSH:
        print("RUN_GITHUB_PUSH=False 이므로 GitHub commit/push는 실행하지 않습니다.")
        return
    if not running_in_colab():
        raise RuntimeError("RUN_GITHUB_PUSH=True는 Colab 실행을 전제로 합니다.")
    from google.colab import userdata

    token = userdata.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("Colab Secrets/Userdata에 GITHUB_TOKEN을 먼저 저장하세요.")
    Path("/root/.netrc").write_text(
        f"machine github.com\nlogin x-access-token\npassword {token}\n",
        encoding="utf-8",
    )
    os.chmod("/root/.netrc", 0o600)

    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", f"https://github.com/{GITHUB_REPO}.git", str(REPO_DIR)], check=True)
    subprocess.run(["git", "switch", BRANCH_NAME], cwd=REPO_DIR, check=False)
    subprocess.run(["git", "add", "analysis/three_axis_clustering"], cwd=REPO_DIR, check=True)
    status = subprocess.run(["git", "status", "--short"], cwd=REPO_DIR, text=True, capture_output=True, check=True)
    if not status.stdout.strip():
        print("No GitHub changes to commit.")
        return
    subprocess.run(["git", "commit", "-m", "Add three-axis clustering analysis results"], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "push", "origin", BRANCH_NAME], cwd=REPO_DIR, check=True)


def publish_reviewed_outputs():
    if LAST_OUTCOME is None or LAST_K_METRICS is None:
        raise RuntimeError("먼저 main()을 실행해서 분석 결과를 만든 뒤 호출하세요.")
    share_dir = build_github_share(LAST_OUTCOME, LAST_K_METRICS)
    maybe_push_to_github()
    return share_dir
