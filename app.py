import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
import re

# --- 설정 ---
# 모델 및 벡터라이저 파일 경로 정의 (실제 경로에 맞게 수정해주세요)
MODEL_PATH = "clickbait_model.pkl"
VEC_PATH = "tfidf_vectorizer.pkl"

# --- Streamlit 페이지 설정 (스크립트의 첫 번째 Streamlit 명령이어야 함!) ---
st.set_page_config(page_title="낚시성 뉴스 판별기", page_icon="🎣", layout="centered")


# --- 모델 및 벡터라이저 로딩 ---
@st.cache_resource
def load_model_and_vectorizer():
    """
    지정된 경로에서 머신러닝 모델과 TF-IDF 벡터라이저를 로드합니다.
    """
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"오류: 필수 파일이 없습니다. 다음 경로를 확인해주세요:")
        st.error(f"- 모델 파일: `{MODEL_PATH}`")
        st.error(f"- 벡터라이저 파일: `{VEC_PATH}`")
        st.stop()
    except Exception as e:
        st.error(f"모델 로드 중 예기치 않은 오류가 발생했습니다: {e}")
        st.stop()

# 앱 시작 시 모델과 벡터라이저를 로드합니다.
model, vectorizer = load_model_and_vectorizer()

# --- 뉴스 링크에서 제목/본문/출처 추출 함수 ---
def extract_info_from_url(url):
    """
    주어진 URL에서 뉴스 기사의 제목, 본문, 출처를 추출합니다.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # --- 인코딩 처리 강화 ---
        # 1. HTTP 헤더의 Content-Type에서 charset을 확인합니다.
        # 2. 또는 특정 도메인(dt.co.kr)에 대해 euc-kr을 강제 적용합니다.
        
        # 기본 인코딩으로 먼저 시도
        text_content = response.text
        
        # Content-Type 헤더에서 charset 정보를 가져오거나, dt.co.kr 도메인이라면 euc-kr 강제 시도
        content_type = response.headers.get('Content-Type', '').lower()
        if 'charset=euc-kr' in content_type or 'dt.co.kr' in url:
            try:
                # response.content (바이트)를 euc-kr로 디코딩
                text_content = response.content.decode('euc-kr', errors='replace') # errors='replace'로 깨진 문자 대체
                st.info("💡 인코딩 문제 감지: `euc-kr`로 재디코딩했습니다.")
            except UnicodeDecodeError:
                st.warning("경고: `euc-kr` 디코딩 중 오류가 발생했지만 진행합니다.")
                pass # euc-kr 디코딩 실패 시 원래 text_content 유지

        soup = BeautifulSoup(text_content, 'html.parser')

        # 1. 제목 추출 (og:title 메타 태그 > title 태그 순으로 시도)
        title_tag = soup.find("meta", property="og:title")
        title = title_tag["content"] if title_tag and title_tag.get("content") else (soup.title.string if soup.title else "제목 없음")
        title = title.strip() if title else "제목 없음"

        # 2. 본문 추출 (다양한 본문 태그/클래스/ID 시도)
        body = ""
        possible_body_selectors = [
            'div.article_view',         # <-- 여기를 가장 위에 추가했습니다!
            'div.article_head_body',    # <-- article_view를 감싸는 부모 태그도 추가 (보험용)
            'div#article_content',      # 기존 디지털타임즈 추정 선택자 (남겨둠)
            'div.article_body_content', # 네이버 뉴스
            'div#harmonyContainer',     # 다음 뉴스
            'div.article-view-content-wrapper',
            'div.news_content',
            'div.article_content',
            'div.body_content',
            'div.entry-content',
            'section.article-body',
            'div.view_page_text',
            'div[itemprop="articleBody"]',
            'article',
            'div#articleBody',
            'div.contents_area',
            'div.news_view',
            'div.view_txt',
            'div.txt_area',
            'div.article_text',
            'div.news_content_area',
            'div.cont_art',
        ]
        
        for selector in possible_body_selectors:
            body_element = None
            if '[' in selector and ']' in selector: # 속성 선택자 (예: div[itemprop="articleBody"])
                body_element = soup.select_one(selector)
            elif '.' in selector: # 클래스 선택자
                tag, class_name = selector.split('.')
                body_element = soup.find(tag, class_=class_name)
            elif '#' in selector: # ID 선택자
                tag, id_name = selector.split('#')
                body_element = soup.find(tag, id=id_name)
            else: # 태그 이름만
                body_element = soup.find(selector)
            
            if body_element:
                # 스크립트, 광고, 주석 등 불필요한 요소 제거
                for s in body_element.find_all(['script', 'style', 'ins', 'iframe', 'noscript', 'img', 'figure', 'ul', 'ol', 'blockquote', 'a', 'strong', 'em']):
                    if s.get_text(strip=True).strip() == '' or s.name in ['script', 'style', 'ins', 'iframe', 'noscript']:
                        s.decompose()
                
                body = body_element.get_text(separator=' ', strip=True)
                body = re.sub(r'\s+', ' ', body).strip()

                if len(body) > 150: # 최소 150자 이상이면 유효하다고 판단
                    break
        
        # 최후의 수단: 만약 위 선택자들로 본문 추출에 실패하면, 모든 <p> 태그의 텍스트를 모으기
        if not body or len(body) < 150:
            all_paragraphs = soup.find_all('p')
            # 짧은 p 태그(예: 저작권, 기자 이름)는 제외하는 필터링 강화
            valid_paragraphs = [p for p in all_paragraphs if len(p.get_text(strip=True)) > 20 and not p.find_parent(class_=['reply', 'footer', 'copyright', 'ad_unit'])]
            if valid_paragraphs:
                temp_body = ' '.join(p.get_text(separator=' ', strip=True) for p in valid_paragraphs)
                temp_body = re.sub(r'\s+', ' ', temp_body).strip()
                if len(temp_body) > 100:
                    body = temp_body
        
        if not body:
            body = "본문 없음"

        # 3. 출처 추출 (URL에서 도메인 이름 추출 개선)
        match = re.search(r"https?://(?:www\.)?([a-zA-Z0-9-]+\.(?:co\.kr|com|net|org|kr|io|dev|ai|app|info|biz|tv|news|me|blog|cc|xyz)[^/]*)/?", url)
        if match and match.group(1):
            domain_full = match.group(1)
            if '.co.kr' in domain_full:
                source = domain_full.split('.')[0]
            else:
                source = domain_full.split('.')[-2]
        else:
            source = "출처 불명"
        
        source = source.replace('-', '').strip()
        if not source: source = "출처 불명"

        return title, body, source

    except requests.exceptions.RequestException as e:
        st.error(f"웹 요청 오류: {e}. URL을 다시 확인하거나 인터넷 연결 상태를 확인해주세요.")
        return "", "", ""
    except Exception as e:
        st.warning(f"링크 분석 중 예기치 않은 오류 발생: {e}. 일부 정보를 가져오지 못할 수 있습니다.")
        return "제목 없음", "본문 없음", "출처 불명"


# --- Streamlit 앱 UI ---
st.header("📰 낚시성 뉴스 판별기")
st.markdown("""
    뉴스 제목, 본문 또는 기사 링크를 입력하면
    자체 학습한 AI가 뉴스의 낚시성 정도를 분석해드립니다.

    
""")

st.markdown("---") # 여기에 첫 번째 구분선이 있습니다.

# 🎨 커스텀 CSS 주입 START
st.markdown("""
<style>
/* 판별하기 버튼 스타일 */
/* primaryColor를 따르도록 type="primary"를 사용하면서 폰트 크기 조절 */
/* Streamlit 1.x 버전에서는 data-testid가 변경될 수 있으므로, 실제 HTML 구조를 확인하여 조정이 필요할 수 있습니다. */
/* 일반적인 primary 버튼의 선택자 */
button[data-testid="stButton"] {
    background-color: #4979ea; /* primaryColor에 이미 설정했지만, 명시적으로 지정 */
    color: white;
    font-size: 1.5em; /* 폰트 크기 키움 (1.5배) */
    padding: 15px 30px; /* 패딩을 늘려 버튼 크기를 키움 */
    border-radius: 8px; /* 모서리를 둥글게 */
    border: none;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* 그림자 효과 추가 */
    transition: background-color 0.3s ease; /* 호버 시 부드러운 전환 */
}

button[data-testid="stButton"]:hover {
    background-color: #3b60c7; /* 호버 시 약간 더 진한 색상 */
}

/* 선택 방식 라디오 버튼의 원형 체크 색상 */
/* Streamlit 라디오 버튼의 HTML 구조는 복잡하므로, 가장 안정적인 선택자를 사용합니다. */
div.stRadio > label > div[data-testid="stCheck"] > div {
    border-color: #4979ea !important; /* 체크박스/라디오 버튼 테두리 색상 */
}
div.stRadio > label > div[data-testid="stCheck"] > div > svg {
    fill: #4979ea !important; /* 체크 표시 또는 원형 내부 색상 */
}

/* 선택 방식 라디오 버튼의 텍스트 색상 (활성 상태) */
div.stRadio > label.st-selected > div > p {
    color: #4979ea !important; /* 선택된 라디오 버튼 텍스트 색상 */
    font-weight: bold; /* 선택된 텍스트 굵게 */
}

/* 선택 방식 라디오 버튼의 텍스트 크기 (필요 시) */
div.stRadio > label > div > p {
    font-size: 1em;
} */


/* 다른 입력 필드 텍스트 크기 (선택 사항) */
/* .stTextArea textarea, .stTextInput input {
    font-size: 1em;
} */

/* 섹션 제목(subheader) 색상 변경 (선택 사항) */
h2 {
    color: #4979ea; /* 섹션 제목 색상 */
}

</style>
""", unsafe_allow_html=True)
# 🎨 커스텀 CSS 주입 END


# 🎣 사이드바 추가 START
with st.sidebar:
    st.header("⚙️ 사이트 정보")
    st.write("이 사이트는 매탄고등학교 학생들이 자극적인 제목이나 낚시성 기사에 쉽게 현혹되지 않고, 올바른 정보 판단 능력을 기를 수 있도록 제작된 사이트로 뉴스 기사의 낚시성 여부를 텍스트 분석을 통해 판별합니다.")
    st.write("**개발자:** GOMHAN") # 여기에 귀하의 정보 추가
    st.write("**버전:** 1.0.1")
    st.markdown("---")
    st.subheader("✔️ 추천 검사 방식")
    st.write("""
    **다음 순서에 따라 검사 정확도가 달라집니다**
    1. 뉴스 기사 링크 입력
    2. 제목 + 본문 입력
    3. 제목만 입력
    """)
    st.markdown("---")
    st.subheader("❓ 사용법")
    st.write("""
    1.  **검사 방식 선택:** 제목만 입력, 제목+본문 입력, 링크 입력 중 하나를 선택하세요.
    2.  **내용 입력:** 선택한 방식에 맞게 텍스트나 링크를 입력합니다.
    3.  **'판별하기' 버튼 클릭:** 모델이 입력된 내용을 분석하고 결과를 보여줍니다.
    """)
    st.markdown("---")
    st.caption("""
    본 사이트는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "낚시성 기사 탐지 데이터"을 활용하여 제작되었습니다. 본 사이트에 활용된 데이터는 AI 허브에서 다운로드 받으실 수 있습니다.
    """)
    st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span>문의 : JH.Moon213@gmail.com</span>
        <img src="https://i.imgur.com/VWthizR.png" width="40" style="margin-left: 10px;" />
    </div>
    """,
    unsafe_allow_html=True
    )

   

# 🎣 사이드바 추가 END

# 📍 메인 콘텐츠 영역 START
col1, col2 = st.columns([1, 2]) # UI 컬럼 분할

with col1: # 첫 번째 컬럼에 검사 방식 선택 배치
    st.subheader("① 검사 방식 선택")
    # 기존 st.radio 코드를 여기에 그대로 둡니다.
    check_method = st.radio(
        "어떤 방식으로 뉴스 내용을 검사하시겠습니까? 👇",
        ("① 제목만 입력", "② 제목 + 본문 입력", "③ 뉴스 기사 링크 입력"),
        index=0, # 기본값은 "제목만 입력"
        key="check_method_radio"
    )
    st.markdown("<br>", unsafe_allow_html=True)

with col2: # 두 번째 컬럼에 입력 필드 배치
    st.subheader("② 정보 입력")
    # 기존 입력 필드 코드를 여기에 그대로 둡니다.
    title_input = ""
    body_input = ""
    link_input = ""
    text_to_analyze = ""
    accuracy_hint = ""

    if check_method == "① 제목만 입력":
        title_input = st.text_area(
            "판별하고 싶은 뉴스 제목 또는 문장을 입력하세요:",
            placeholder="예: '이것만 알면 당신도 부자! 비밀 공개'",
            height=100,
            key="title_only_input"
        )
        accuracy_hint = "정확도: 낮음 (제목만 사용)"
        text_to_analyze = title_input

    elif check_method == "② 제목 + 본문 입력":
        title_input = st.text_input(
            "뉴스 제목을 입력하세요:",
            placeholder="예: '이것만 알면 당신도 부자! 비밀 공개'",
            key="title_and_body_title_input"
        )
        body_input = st.text_area(
            "뉴스 본문을 입력하세요:",
            placeholder="기사 본문 내용을 여기에 붙여넣으세요.",
            height=200,
            key="title_and_body_body_input"
        )
        accuracy_hint = "보통: 높음 (제목 + 본문 사용)"

    elif check_method == "③ 뉴스 기사 링크 입력":
        link_input = st.text_input(
            "뉴스 기사 링크(URL)를 입력하세요:",
            placeholder="예: https://news.naver.com/main/read.naver?mode=LSD&mid=shm&sid1=105&oid=001&aid=0012345678",
            key="link_input"
        )
        accuracy_hint = "정확도: 높음 (제목 + 본문 + 출처 사용)"

st.markdown("---")

# ✅ 판별하기 버튼 START
# 기존 'if st.button("판별하기", key="predict_button"):' 라인을 아래 코드로 교체합니다.
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    if st.button("✨ 판별하기 ✨", key="predict_button", use_container_width=True, type="primary"):
        # `type="primary"`를 사용하면 `config.toml`의 `primaryColor`를 따릅니다.
        # `config.toml`로 색상을 변경했으므로, CSS에서는 크기, 그림자 등만 제어하면 됩니다.
        
        # 입력값 검증 및 분석할 텍스트 준비
        if check_method == "① 제목만 입력":
            if not title_input.strip():
                st.warning("제목을 입력해주세요. 비어있습니다.")
                st.stop()
            text_to_analyze = title_input

        elif check_method == "② 제목 + 본문 입력":
            if not title_input.strip() and not body_input.strip():
                st.warning("제목과 본문 중 하나라도 입력해주세요.")
                st.stop()
            
            if title_input.strip() and not body_input.strip():
                text_to_analyze = title_input
                accuracy_hint = "정확도: 낮음 (제목만 사용)"
            elif not title_input.strip() and body_input.strip():
                text_to_analyze = body_input
                accuracy_hint = "정확도: 보통 (본문만 사용)"
            else:
                text_to_analyze = title_input + " " + body_input
                accuracy_hint = "정확도: 높음 (제목 + 본문 사용)"

        elif check_method == "③ 뉴스 기사 링크 입력":
            if not link_input.strip():
                st.warning("뉴스 기사 링크를 입력해주세요. 비어있습니다.")
                st.stop()
            
            with st.spinner('🔗 링크에서 뉴스 정보 추출 중... (최대 10초)'):
                title_extracted, body_extracted, source_extracted = extract_info_from_url(link_input)
            
            if not title_extracted and not body_extracted:
                st.error("❌ 링크에서 내용을 불러오지 못했습니다. 링크를 다시 확인하거나 다른 링크를 시도해주세요.")
                st.stop()
            elif title_extracted == "제목 없음" and body_extracted == "본문 없음":
                st.error("⚠️ 링크에서 유효한 제목과 본문을 찾을 수 없습니다. 뉴스 기사가 맞는지 확인해주세요.")
                st.stop()
            else:
                text_to_analyze = f"{title_extracted} {body_extracted} {source_extracted}"
                st.success(f"✅ **추출된 제목:** {title_extracted if title_extracted != '제목 없음' else '제목 없음'}")
                st.info(f"**추출된 출처:** {source_extracted}")
                if body_extracted != "본문 없음":
                    with st.expander("📝 추출된 본문 미리보기"): # 본문 미리보기를 Expander로 감싸기
                        st.write(body_extracted[:500] + "..." if len(body_extracted) > 500 else body_extracted)
                else:
                    st.warning("⚠️ 본문 추출에 실패했습니다. 제목과 출처 정보만으로 분석합니다.")

        # 최종 분석할 텍스트가 비어있다면 오류 처리
        if not text_to_analyze.strip():
            st.warning("분석할 텍스트를 준비하는 데 실패했습니다. 다시 시도해주세요.")
            st.stop()

        # --- 모델 예측 ---
        with st.spinner("🧠 모델이 낚시성 여부를 분석 중입니다..."):
            X_vec = vectorizer.transform([text_to_analyze])
            
            probabilities = model.predict_proba(X_vec)[0]
            
            try:
                clickbait_class_index = list(model.classes_).index(1) # 1이 낚시성이라는 전제
                prob_clickbait = probabilities[clickbait_class_index]
                percent_clickbait = round(prob_clickbait * 100, 2)
            except ValueError:
                st.error("오류: 모델 클래스에 낚시성/정상 라벨(0 또는 1)이 정의되어 있지 않습니다. 모델 학습을 확인하세요.")
                st.stop()

            predicted_label = model.predict(X_vec)[0]

        # --- 결과 출력 ---
        st.markdown("---")
        st.subheader("📊 판별 결과" )
        st.markdown("<br>", unsafe_allow_html=True)
        # 결과 메시지를 강조하고 이모지로 시각화
        if predicted_label == 1: # 모델이 낚시성(1)으로 예측한 경우
            st.markdown(
            f"<p style='font-size:17px;'>🚨 이 기사는 <strong>낚시성 뉴스</strong>일 확률이 <strong>{percent_clickbait}%</strong> 입니다!</p>",
            unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.error("❗ **높은 확률로 독자의 클릭을 유도하는 요소를 포함하고 있습니다.**")
            if percent_clickbait > 70:
                st.caption("주의! 자극적인 표현이나 과장된 내용이 있을 수 있습니다.")
            else:
                st.caption("낚시성으로 예측되었으나 확률은 다소 낮습니다. (모델의 판단)")
        elif predicted_label == 0: # 모델이 정상(0)으로 예측한 경우
            st.markdown(f"## ✅ 이 기사는 **정상 뉴스**일 확률이 `{100 - percent_clickbait}%` 입니다.")
            st.success("👍 **낚시성 특징이 거의 없는 일반적인 뉴스입니다.**")
            if percent_clickbait < 30:
                 st.caption("안심하고 읽으셔도 좋습니다.")
            else:
                st.caption("낚시성으로 예측되지 않았지만, 확률이 애매할 수 있습니다. (판단 기준: 낚시성 확률 40% 이하)")
        else:
            st.warning(f"알 수 없는 라벨 값({predicted_label})이 예측되었습니다. 데이터 라벨링을 확인하세요.")
        
        st.info(accuracy_hint) # 정확도 힌트도 여기에 표시

st.markdown("---")
st.markdown("문의 : JH.Moon213@gmail.com")
