import streamlit as st
import pickle
import re
import nltk
import os
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
import PyPDF2
import pandas as pd
from io import BytesIO
import base64
#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app
def process_resume(resume_text):
    cleaned_resume = clean_resume(resume_text)
    input_features = tfidfd.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    probabilities = clf.predict_proba(input_features)[0]  # This assumes that input_features is for one sample
    top_three_indices = np.argsort(probabilities)[-3:][::-1]
    # Map category ID to category name
    category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
    predictions = [(category_mapping.get(index, "Unknown"), round(probabilities[index] * 100, 2)) for index in top_three_indices]
    return predictions

def standardize_resume_file(uploaded_file):
    
    if uploaded_file is not None:
        resume_text = ""
        if uploaded_file.type == 'text/plain':
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')
        elif uploaded_file.type == 'application/pdf':
            with st.spinner('Converting PDF to text...'):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    resume_text += page.extract_text()
        else:
            st.error("Unsupported file format. Please upload a TXT or PDF file.")
            return None
        return resume_text

def extract_top_1(resume_text):
    cleaned_resume = clean_resume(resume_text)
    input_features = tfidfd.transform([cleaned_resume])
    probabilities = clf.predict_proba(input_features)[0]
    top_category_index = np.argmax(probabilities)
    
    # Map category ID to category name
    category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
    
    top_category = category_mapping.get(top_category_index, "Unknown")
    top_probability = round(probabilities[top_category_index] * 100, 2)
    
    return [(top_category, top_probability)]

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

def main():
    # các categories để lựa chọn
    categories = [
        "Java Developer", "Testing", "DevOps Engineer", "Python Developer", "Web Designing",
        "HR", "Hadoop", "Blockchain", "ETL Developer", "Operations Manager", 
        "Data Science", "Sales", "Mechanical Engineer", "Arts", "Database", 
        "Electrical Engineering", "Health and fitness", "PMO", "Business Analyst", 
        "DotNet Developer", "Automation Testing", "Network Security Engineer", "SAP Developer", 
        "Civil Engineer", "Advocate"
    ]
    st.title("Resume Screening App")
    
    
    # Di chuyển checkbox sang bên phải màn hình
    with st.sidebar:
        st.write("## Categories")
        selected_categories = []
        for category in categories:
            if st.checkbox(category):
                selected_categories.append(category)

    # Tạo danh sách tên upload_files để upload nhiều file cùng một lúc
    upload_files = st.file_uploader('Upload Resumes', type=['txt', 'pdf'], accept_multiple_files=True, key="file_uploader")
    if upload_files is not None:
        for uploaded_file in upload_files:
            st.write(f"File Name: {uploaded_file.name}")
            # Hiển thị tên của từng file được upload
            st.write(f"Content Type: {uploaded_file.type}")
            # Hiển thị loại nội dung của file (ví dụ: text/plain, application/pdf)
            st.write(f"File Size: {uploaded_file.size} bytes")
            # Hiển thị kích thước của file

            resume_text = standardize_resume_file(uploaded_file)
            if resume_text is not None:
                st.write("Resume Content:")
                st.text(resume_text)  # Hiển thị nội dung của resume
                predictions = process_resume(resume_text)
                # top_1_tier = extract_top_1(resume_text)
                # top_1_category, top_1_probability = top_1_tier[0]
                st.write(f"Top 3 Predicted Categories for {uploaded_file.name}:")
                for category, probability in predictions:
                    st.write(f"{category}: {probability}%")

    if selected_categories:
        selected_resumes = []
        for uploaded_file in upload_files:
            resume_text = standardize_resume_file(uploaded_file)
            if resume_text is not None:
                predictions = process_resume(resume_text)
                top_1_tier = extract_top_1(resume_text)
                top_1_category, _ = top_1_tier[0]
                if top_1_category in selected_categories:
                    selected_resumes.append((top_1_category, resume_text))
        if selected_resumes:
            # Tạo DataFrame từ danh sách các CV được lựa chọn
            df = pd.DataFrame(selected_resumes, columns=["Category", "Resume"])
            st.dataframe(df)
            # Tạo đối tượng BytesIO từ DataFrame
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)

            # Gửi dữ liệu đến người dùng với loại tệp tin là 'xlsx'
            excel_data = excel_buffer.getvalue()
            b64 = base64.b64encode(excel_data).decode('utf-8')
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="selected_resumes.xlsx">Download Selected Resumes as Excel</a>'
            st.markdown(href, unsafe_allow_html=True)
    # if uploaded_file is not None:
    #     resume_text = standardize_resume_file(uploaded_file)
    #     if resume_text is not None:
    #         predictions = process_resume(resume_text)
    #         st.write("Top 3 Predicted Categories:")
    #         for category, probability in predictions:
    #             st.write(f"{category}: {probability}%")
    
# python main
if __name__ == "__main__":
    main()







# def main():
#     st.title("Resume Screening App")
#     uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

#     if uploaded_file is not None:
#         try:
#             resume_bytes = uploaded_file.read()
#             resume_text = resume_bytes.decode('utf-8')
#         except UnicodeDecodeError:
#             # If UTF-8 decoding fails, try decoding with 'latin-1'
#             resume_text = resume_bytes.decode('latin-1')

#         cleaned_resume = clean_resume(resume_text)
#         input_features = tfidfd.transform([cleaned_resume])
#         prediction_id = clf.predict(input_features)[0]

#         probabilities = clf.predict_proba(input_features)[0]  # This assumes that input_features is for one sample
#         # print(probabilities) #in xem thông số của mỗi cái trong cái mapping categories

#         # Get the indices of the top 3 probabilities
#         top_three_indices = np.argsort(probabilities)[-3:]

#         # The indices are in ascending order of probabilities, so you might want to reverse them:
#         top_three_indices = top_three_indices[::-1]

#         # Map category ID to category name
#         category_mapping = {
#             15: "Java Developer",
#             23: "Testing",
#             8: "DevOps Engineer",
#             20: "Python Developer",
#             24: "Web Designing",
#             12: "HR",
#             13: "Hadoop",
#             3: "Blockchain",
#             10: "ETL Developer",
#             18: "Operations Manager",
#             6: "Data Science",
#             22: "Sales",
#             16: "Mechanical Engineer",
#             1: "Arts",
#             7: "Database",
#             11: "Electrical Engineering",
#             14: "Health and fitness",
#             19: "PMO",
#             4: "Business Analyst",
#             9: "DotNet Developer",
#             2: "Automation Testing",
#             17: "Network Security Engineer",
#             21: "SAP Developer",
#             5: "Civil Engineer",
#             0: "Advocate",
#         }

#         # category_name = category_mapping.get(prediction_id, "Unknown")
#         # st.write("Predicted Category:", category_name)

#         # Print top 3 predicted categories and their probabilities
#         for index in top_three_indices:
#             category_name = category_mapping.get(index, "Unknown")
#             probability_percent = round(probabilities[index] * 100, 2)
#             st.write(f"{category_name}: {probability_percent}%")