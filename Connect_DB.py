#DB랑 단순 연결 확인(URL/KEY)
#나중에 db랑 값 주고받는걸로 바꿔야함 
from supabase import create_client, Client

# 여기에 본인의 Supabase 정보 입력
SUPABASE_URL = "https://hriojgloycvfguijvbli.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhyaW9qZ2xveWN2Zmd1aWp2YmxpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0OTE1MTksImV4cCI6MjA2ODA2NzUxOX0.htJrKNzPW1E752zMwEAWB0TmX9CUboin0biV0vA8pX0"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 연결 테스트: '테이블 이름'이 있다면 데이터를 가져오고, 없다면 에러 메시지 출력
try:
    response = supabase.table("accupancy_logs").select("*").execute()
    print("✅ 연결 성공!")
    print(response.data)
    
except Exception as e:
    print("❌ 연결 실패 또는 테이블 없음")
    print(e)
