"""Demo using the same locked artifacts and inference as the evaluator."""
import os
import sys
from pathlib import Path

import streamlit as st
import torch
import yaml
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))
from liver_ct import load_artifact, pipeline_labels, predict_images, validate_pair

st.set_page_config(page_title='Liver CT Image Classification', page_icon='🩺', layout='centered')
st.title('Liver CT Image Classification Using Prototypical Networks')
st.caption('Phân loại ảnh thành: không có u, u lành tính hoặc u ác tính.')


@st.cache_resource
def load_pipeline(presence_path, subtype_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pmodel, presence = load_artifact(presence_path, device)
    smodel, subtype = load_artifact(subtype_path, device)
    validate_pair(presence, subtype)
    return pmodel, presence, smodel, subtype, device


config_path = Path(os.environ.get('LIVER_CT_CONFIG', str(APP_DIR / 'config.yaml')))
try:
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    base = config_path.resolve().parent
    pmodel, presence, smodel, subtype, device = load_pipeline(
        str((base / config['presence_artifact']).resolve()),
        str((base / config['subtype_artifact']).resolve()))
except (OSError, KeyError, ValueError, RuntimeError, TypeError, yaml.YAMLError):
    st.info('Chưa có bộ mô hình hợp lệ để phân loại. Vui lòng cấu hình mô hình đã được đánh giá.')
    st.stop()

if presence.get('evidence_status') == 'INTEGRATION_ONLY':
    st.warning('Bộ mô hình này chỉ dùng kiểm tra kỹ thuật; kết quả chưa được đánh giá trên dữ liệu thực.')
elif presence.get('ancestry_status') == 'UNKNOWN':
    st.warning('Mô hình thăm dò trên dữ liệu ảnh chưa xác định bệnh nhân và ảnh gốc; kết quả chưa được kiểm chứng lâm sàng.')
elif presence.get('label_provenance') == 'UNVERIFIED':
    st.warning('Nhãn của dữ liệu huấn luyện chưa được kiểm chứng; kết quả chỉ mang tính thăm dò.')

files = st.file_uploader('Chọn tối đa 2 ảnh CT', type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'],
                         accept_multiple_files=True)
images = []
for uploaded in (files or [])[:2]:
    try:
        with Image.open(uploaded) as image:
            images.append(image.convert('RGB').copy())
    except OSError:
        st.error(f'Không đọc được ảnh: {uploaded.name}')
if files and len(files) > 2:
    st.info('Chỉ xử lý 2 ảnh đầu tiên.')
if images:
    columns = st.columns(len(images))
    for column, image in zip(columns, images):
        column.image(image, use_container_width=True)
    if st.button('Phân loại', use_container_width=True):
        try:
            p1 = predict_images(pmodel, presence, images, device)
            p2 = predict_images(smodel, subtype, images, device)
            labels = pipeline_labels(p1, p2, presence['threshold'], subtype['threshold'])
            names = ['Không có u', 'U lành tính', 'U ác tính']
            for column, label in zip(columns, labels):
                column.write(f'Kết quả dự đoán: **{names[label]}**')
        except (ValueError, RuntimeError):
            st.error('Không thể phân loại bộ ảnh này. Vui lòng kiểm tra ảnh và bộ mô hình.')
