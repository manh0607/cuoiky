from preprocess import text_preprocess, remove_html
from utils import remove_html, remove_stopwords
# xem kết quả cho 1 văn bản model naive bayes đã load ở trên
import numpy as np
import pickle
import os
# Chia tập train/test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
test_percent = 0.2

# MODEL_PATH='models'

text = []
label = []

for line in open('news_categories.prep','r', encoding='utf-8'):
    words = line.strip().split()
    label.append(words[0])
    text.append(' '.join(words[1:]))

X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)

# # Lưu train/test data
# # Giữ nguyên train/test để về sau so sánh các mô hình cho công bằng
with open('train.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_train, y_train):
        fp.write('{} {}\n'.format(y, x))

with open('test.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_test, y_test):
        fp.write('{} {}\n'.format(y, x))

# # encode label
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print(list(label_encoder.classes_), '\n')
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# print(X_train[0], y_train[0], '\n')
# print(X_test[0], y_test[0])
# Naive Bayes
model = pickle.load(open("naive_bayes.pkl", 'rb'))
# model.eval()
# y_pred = model.predict(X_test)
# print('Naive Bayes, Accuracy =', np.mean(y_pred == y_test))

# Linear Classifier
# model = pickle.load(open(os.path.join(MODEL_PATH,"linear_classifier.pkl"), 'rb'))
# y_pred = model.predict(X_test)
# print('Linear Classifier, Accuracy =', np.mean(y_pred == y_test))
    
# # print_results(*model.test('test.txt'))
# document = "di đá banh và bị công an bắt"
 
# document = text_preprocess(document)
# document = remove_stopwords(document)

# label = model.predict([document])
# print('Predict label:', label_encoder.inverse_transform(label))

def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + ''.join(i.capitalize() for i in s[1:])

st.set_page_config(page_title='test')
st.title('Phân Loại Tin Tức')
with st.form("my_form"):
    user_input = st.text_area("Nhập văn bản")
    submitted = st.form_submit_button('Phân Loại')
    if submitted:
        if(user_input == ''):
            st.error('Vui lòng nhập văn bản !')
        else:
            user_input = text_preprocess(user_input)
            user_input = remove_stopwords(user_input)
            label = model.predict([user_input])
            result = str(label_encoder.inverse_transform(label))
            result = result.replace('[', '').replace(']', '').replace("'", "").strip()
            st.write("Kết quả ")
            st.success(result[9:].replace('_', ' '))