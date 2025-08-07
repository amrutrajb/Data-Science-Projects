# app.py
import streamlit as st
import tempfile, cv2, os, re, easyocr, numpy as np, pandas as pd, io
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from difflib import SequenceMatcher

st.set_page_config("Number Plate Detector", layout="wide")
st.title("🚗 Automatic Number Plate Recognition")
st.write("Upload an MP4 video to detect and consolidate number plates.")

uploaded = st.file_uploader("Upload MP4", type=["mp4"])
if not uploaded:
    st.info("Please upload an MP4 video to get started.")
    st.stop()

tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded.read())
tfile.close()

model = YOLO(r'D:\Downloads\InsightFlow AI\am\300_EPOCH_BEST_NPR.pt')
reader = easyocr.Reader(['en'], gpu=False)

pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$')
corr = {'6':'G','4':'A','1':'A'}
str_corr = {'TH':'MH','WH':'MH','NH':'MH','TW':'MH','AH':'MH'}

cap = cv2.VideoCapture(tfile.name)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
padding, frame_idx = 10, 0
records = []

while True:
    ok, im0 = cap.read()
    if not ok: break
    frame_idx += 1
    res = model.predict(im0, imgsz=640, conf=0.3)[0]
    anns = Annotator(im0, line_width=2)

    for cls, box, conf in zip(res.boxes.cls.cpu(), res.boxes.xyxy.cpu(), res.boxes.conf.cpu()):
        x1,y1,x2,y2 = map(int, box)
        crop = im0[max(y1-padding,0):min(y2+padding,h), max(x1-padding,0):min(x2+padding,w)]
        ocr = reader.readtext(crop, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if not ocr: continue
        s = ''.join(ocr).replace(' ', '').upper()
        if len(s)==10:
            for k,v in str_corr.items():
                if s.startswith(k): s = v + s[2:]
            lst = list(s)
            for pos in [0,1,4,5]:
                if lst[pos] in corr: lst[pos] = corr[lst[pos]]
            s = ''.join(lst)
            if pattern.match(s):
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
                records.append({'frame': frame_idx, 'plate': s, 'timestamp': ts, 'confidence': float(conf)})
                anns.box_label(box, label=s, color=colors(int(cls), True))
    cap_img = anns.result()  # Optionally display: st.image(cap_img)

cap.release()
os.unlink(tfile.name)

if not records:
    st.warning("No valid plates detected.")
    st.stop()

df = pd.DataFrame(records)

def similar(a,b,th=0.5):
    return SequenceMatcher(None, a, b).ratio() >= th

final = []
used = set()
for idx, row in df.iterrows():
    if idx in used: continue
    group = df[df['plate'].apply(lambda x: similar(x, row['plate'])) & (~df.index.isin(used))]
    used.update(group.index)
    best = group.loc[group['confidence'].idxmax()]
    final.append(best)

final_df = pd.DataFrame(final).reset_index(drop=True)

st.sidebar.header("Filter & Download")
plates = ["-- All Plates --"] + sorted(final_df['plate'].unique().tolist())
choice = st.sidebar.selectbox("Choose Plate:", plates)
df_show = final_df if choice == "-- All Plates --" else final_df[final_df['plate']==choice]

st.dataframe(df_show, use_container_width=True)

@st.cache_data
def to_csv(data): return data.to_csv(index=False).encode('utf-8')
@st.cache_data
def to_xlsx(data):
    buf = io.BytesIO()
    data.to_excel(buf, index=False, engine='xlsxwriter')
    return buf.getvalue()

st.sidebar.download_button("📥 Download CSV", to_csv(df_show), "plates.csv", "text/csv")
st.sidebar.download_button("📥 Download Excel", to_xlsx(df_show), "plates.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.sidebar.success("✅ Processing complete!")
