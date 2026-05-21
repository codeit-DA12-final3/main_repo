# Votes + Hackle visualization results

이 폴더는 Colab에서 생성한 공유용 결과물입니다. 원본 raw CSV는 포함하지 않았습니다.

## Analysis Notes
### 1. votes 시계열 지표
- 조건: 출석, 결제, 질문 기록, 친구 요청을 날짜 단위로 집계
- 의미: votes DB에서 주요 행동이 시간에 따라 언제 많아졌는지 비교합니다.

### 2. votes 결제 시계열
- 조건: accounts_paymenthistory.created_at 기준 일별 결제 건수와 결제 유저 수
- 의미: 결제 활동이 특정 날짜에 집중되는지 확인합니다.

### 3. votes 질문 기록 시계열
- 조건: accounts_userquestionrecord.created_at 기준 일별 질문 기록 수, 열람 수, opened_times 합계
- 의미: 질문 생성/열람 행동의 일별 흐름을 봅니다.

### 4. hackle 시계열 지표
- 조건: hackle_events 단일 테이블에서 event_datetime 기준 일별 이벤트 수와 세션 수
- 의미: 앱 이벤트 로그의 전체 사용량 흐름을 session 기준으로 확인합니다.

### 5. 재방문 1: votes 출석 기반 재방문
- 조건: accounts_attendance.attendance_date_list의 서로 다른 날짜 수가 2일 이상이면 재방문으로 분류
- 의미: 출석 기록 기준으로 1일 출석 유저와 2일 이상 출석 유저의 규모를 비교합니다.

### 6. 재방문 2: hackle session 기준 반복 행동
- 조건: session_id+날짜 기준으로 click_attendance 2회 이상/미만, 전체 로그 2회 이상/미만을 비교
- 의미: 사람 기준이 아니라 세션-일 기준으로 반복 행동이 많은 세션을 봅니다.

### 7. 결제 1: 결제 유저/미결제 유저/평균 결제 횟수
- 조건: accounts_user 전체 user_id 중 accounts_paymenthistory에 등장한 유저를 결제 유저로 분류
- 의미: 결제를 한 번이라도 한 유저 규모와 유저당 평균 결제 빈도를 봅니다.

### 8. 결제 2: productId별 결제 분포
- 조건: productId별 총 결제 건수, 서로 다른 결제 유저 수, 같은 유저가 같은 상품을 2회 이상 결제한 쌍 수
- 의미: 어떤 상품이 많이 팔렸고, 같은 상품을 반복 구매한 유저가 얼마나 있는지 봅니다.

### 9. 자발적 재참여 1: 알림/타임라인 관련 event_key별 세션 분포
- 조건: hackle_events에서 알림/타임라인 관련 event_key만 선택하고, session_id별 event_count 분포를 비교
- 의미: 각 event_key가 세션에서 얼마나 반복적으로 발생하는지 봅니다. click_notice_detail이 알림 클릭 후 상세 진입에 가장 가까운 지표로 추정됩니다.

### 10. 자발적 재참여 2: event_key 전체 발생 건수
- 조건: 선택한 알림/타임라인 event_key별 전체 event_count 합계
- 의미: 어떤 진입/클릭 행동이 가장 자주 발생했는지 비교합니다.

### 11. 자발적 재참여 3: 같은 session에서 다른 날짜에 2일 이상 발생
- 조건: session_id+event_key 기준으로 서로 다른 event_date가 2일 이상이면 2일 이상 그룹으로 분류
- 의미: 같은 세션 식별자에서 특정 알림/타임라인 행동이 날짜를 달리해 반복되는지 봅니다.

### 12. 누적 활동일 수 1: 비연속 활동일 수
- 조건: accounts_attendance.attendance_date_list에서 서로 다른 날짜 수를 user_id별로 계산
- 의미: 한 유저가 전체 기간 중 며칠 활동했는지 분포를 봅니다.

### 13. 누적 활동일 수 2: 최장 연속 활동일 수
- 조건: attendance_date_list에서 하루 차이로 이어지는 최장 연속 출석 구간을 user_id별로 계산
- 의미: 끊기지 않고 연속적으로 활동한 기간이 긴 유저가 얼마나 있는지 봅니다.

### 14. 핵심 행동 반복 1: votes 질문 기록 기준
- 조건: accounts_userquestionrecord에서 user_id 기준과 chosen_user_id 기준으로 record_count, has_read 합계, opened_times 합계를 비교
- 의미: 질문을 던진/받은 역할별로 질문 수와 열람 행동이 어떻게 다른지 봅니다.

### 15. 핵심 행동 반복 2: hackle session 기준 질문 열람
- 조건: session_id별 votes_count 최대값과 click_question_open 발생 횟수를 비교
- 의미: 세션 기준으로 받은 투표/질문 규모와 질문 열람 행동이 같이 움직이는지 봅니다.

### 16. 추천/공유 1: votes 친구 요청
- 조건: accounts_friendrequest에서 send_user_id별 서로 다른 receive_user_id 수와 receive_user_id별 서로 다른 send_user_id 수를 비교
- 의미: 친구 요청을 많이 보낸 유저와 많이 받은 유저의 분포를 봅니다.

### 17. 추천/공유 2: hackle session 기준 초대/공유 이벤트
- 조건: session_id별 click_invite_friend, click_friend_invite, click_question_share 발생 횟수
- 의미: 세션 단위로 친구 초대/질문 공유 행동이 얼마나 발생하는지 봅니다.

## Figures
- `figures/01_votes_attendance_daily.png`
- `figures/02_votes_payment_daily.png`
- `figures/03_votes_question_daily.png`
- `figures/04_hackle_daily.png`
- `figures/05_revisit_votes_attendance.png`
- `figures/06_revisit_hackle_session_day.png`
- `figures/07_payment_paid_unpaid.png`
- `figures/08_payment_product_distribution.png`
- `figures/09_notice_event_distribution.png`
- `figures/10_notice_event_total.png`
- `figures/11_notice_event_2plus_days.png`
- `figures/12_activity_days_bucket.png`
- `figures/13_activity_streak_bucket.png`
- `figures/14_core_question_role_mean.png`
- `figures/15_core_votes_open_heatmap.png`
- `figures/16_core_question_open_bucket.png`
- `figures/17_share_friend_request_bucket.png`
- `figures/18_share_hackle_event_total.png`
- `figures/19_share_hackle_distribution.png`

## Table previews
- `tables/01_빠른_로드_확인_hackle_선택_event_key_요약.csv`
- `tables/02_votes_출석_일별_표.csv`
- `tables/03_votes_결제_일별_표.csv`
- `tables/04_votes_질문_기록_일별_표.csv`
- `tables/05_hackle_일별_표.csv`
- `tables/06_votes_출석_재방문_요약.csv`
- `tables/07_hackle_재방문_반복_행동_요약.csv`
- `tables/08_결제_KPI_표.csv`
- `tables/09_productId별_결제_분포_표.csv`
- `tables/10_알림_타임라인_event_key_분포_요약.csv`
- `tables/11_알림_타임라인_event_key_전체_건수.csv`
- `tables/12_알림_타임라인_2일_이상_반복_세션_표.csv`
- `tables/13_비연속_누적_활동일_수_구간_요약.csv`
- `tables/14_최장_연속_활동일_수_구간_요약.csv`
- `tables/15_질문_기록_역할별_요약.csv`
- `tables/16_hackle_click_question_open_분포_요약.csv`
- `tables/17_친구_요청_구간별_요약.csv`
- `tables/18_친구_요청_분포_요약.csv`
- `tables/19_hackle_초대_공유_요약.csv`
- `tables/20_hackle_초대_공유_session_분포_요약.csv`

## Summary CSV
- `summary_csv/hackle_daily.csv`
- `summary_csv/votes_attendance_daily.csv`
- `summary_csv/votes_friend_daily.csv`
- `summary_csv/votes_payment_daily.csv`
- `summary_csv/votes_product_summary.csv`
- `summary_csv/votes_question_daily.csv`
