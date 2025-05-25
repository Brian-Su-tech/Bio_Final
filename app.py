import streamlit as st
import torch
import esm
import numpy as np

# 設置頁面
st.set_page_config(page_title="SNARE 蛋白質預測", page_icon="🧬")
st.title("SNARE 蛋白質預測系統")

# 載入模型
@st.cache_resource
def load_model():
    try:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        model = model.eval()
        return model, alphabet
    except Exception as e:
        st.error(f"模型載入錯誤：{str(e)}")
        return None, None

# 預測函數
def predict_sequence(sequence, model, alphabet):
    try:
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
        
        # 簡單的預測邏輯（這裡需要根據您的實際模型調整）
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
sequence = st.text_area("序列", height=150)

if st.button("預測"):
    if sequence:
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
    else:
        st.warning("請輸入蛋白質序列")

# 添加說明
st.markdown("""
### 使用說明
1. 在文本框中輸入蛋白質序列
2. 點擊"預測"按鈕
3. 等待預測結果

### 注意事項
- 序列只能包含標準氨基酸代碼（A-Z）
- 預測結果僅供參考
""")
