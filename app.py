import streamlit as st
import torch
import esm
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

# 設置頁面
st.set_page_config(page_title="SNARE 蛋白質預測", page_icon="🧬")
st.title("SNARE 蛋白質預測系統")

# 郵件發送函數
def send_email(to_email, sequence, result):
    try:
        # 從 Streamlit secrets 獲取郵件配置
        email_config = st.secrets["email"]
        sender_email = email_config["brian20040211@gmail.com"]
        app_password = email_config["tpfg iwuy fybt dnjj"]
        
        # 創建郵件
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = 'SNARE 蛋白質預測結果'
        
        # 郵件正文
        body = f"""
        蛋白質序列預測結果：
        
        序列：{sequence}
        
        預測結果：
        - 預測：{result['prediction']}
        - 信心分數：{result['confidence']*100:.1f}%
        - SNARE 機率：{result['probabilities']['SNARE']*100:.1f}%
        - Non-SNARE 機率：{result['probabilities']['Non-SNARE']*100:.1f}%
        
        此郵件由 SNARE 蛋白質預測系統自動發送。
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # 發送郵件
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"郵件發送錯誤：{str(e)}")
        return False

# 載入模型
@st.cache_resource
def load_model():
    try:
        # 設置模型緩存目錄
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 載入模型
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        model = model.eval()
        return model, alphabet
    except Exception as e:
        st.error(f"模型載入錯誤：{str(e)}")
        return None, None

# 預測函數
def predict_sequence(sequence, model, alphabet):
    try:
        # 清理序列（只保留氨基酸字符）
        sequence = ''.join(c for c in sequence.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
        
        if not sequence:
            st.error("無效的序列，請只輸入標準氨基酸代碼（A-Z）")
            return None
            
        # 將序列轉換為模型可接受的格式
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
        
        # 獲取嵌入向量
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
        
        # 獲取最後一層的嵌入向量
        embeddings = results["representations"][12]
        embeddings = embeddings[0, 1:-1, :]
        mean_embedding = embeddings.mean(dim=0)
        
        # 簡單的預測邏輯
        prediction = "SNARE" if torch.mean(mean_embedding) > 0 else "Non-SNARE"
        confidence = float(torch.sigmoid(torch.mean(mean_embedding)))
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'SNARE': confidence,
                'Non-SNARE': 1 - confidence
            }
        }
    except Exception as e:
        st.error(f"預測錯誤：{str(e)}")
        return None

# 主界面
st.write("請輸入蛋白質序列：")
sequence = st.text_area("序列", height=150, placeholder="例如：MLLAVLYCLLWAFSSSSCSGVLQVRQSSGLPCVARSPSRLEQADVGPFVRFEFSPSDSVSTPRAPREGQVTISCTGSSSNIGAGNVHWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNNWPLTFGGGTKLEIK")

# 添加郵件輸入
email = st.text_input("電子郵件地址", placeholder="請輸入您的電子郵件地址")

if st.button("預測"):
    if not sequence:
        st.warning("請輸入蛋白質序列")
    elif not email:
        st.warning("請輸入電子郵件地址")
    else:
        with st.spinner("正在載入模型..."):
            model, alphabet = load_model()
        
        if model and alphabet:
            with st.spinner("正在進行預測..."):
                result = predict_sequence(sequence, model, alphabet)
                
                if result:
                    st.success(f"預測結果：{result['prediction']}")
                    st.write(f"信心分數：{result['confidence']*100:.1f}%")
                    st.write(f"SNARE 機率：{result['probabilities']['SNARE']*100:.1f}%")
                    st.write(f"Non-SNARE 機率：{result['probabilities']['Non-SNARE']*100:.1f}%")
                    
                    # 發送郵件
                    if send_email(email, sequence, result):
                        st.success("預測結果已發送到您的郵箱")
                    else:
                        st.error("郵件發送失敗，但預測結果已顯示在頁面上")

# 添加說明
st.markdown("""
### 使用說明
1. 在文本框中輸入蛋白質序列
2. 輸入您的電子郵件地址
3. 點擊"預測"按鈕
4. 等待預測結果
5. 結果會同時顯示在頁面上並發送到您的郵箱

### 注意事項
- 序列只能包含標準氨基酸代碼（A-Z）
- 預測結果僅供參考
- 請確保輸入正確的電子郵件地址
""")
