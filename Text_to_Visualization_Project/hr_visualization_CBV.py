import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import re

class HRVisualizerApp:
    def __init__(self):
        self.df = pd.read_csv(r'D:\Downloads\InsightFlow AI\venv\Scripts\programs\HR_data_15.csv')
        self.columns = ['EmpID', 'Age', 'AgeGroup', 'Attrition', 'BusinessTravel', 'DailyRate',
                        'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
                        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
                        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
                        'SalarySlab', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime']

    def make_prompt(self, chart_description):
        return f"""
You are a Python data analyst who knows very good python code.

You are working with a pandas DataFrame named `df` that has the following columns:
{', '.join(self.columns)}

The user wants this chart: "{chart_description}".

Generate clean Python code using seaborn. Do NOT include any explanations.
The code must work with the DataFrame `df` already loaded.
Always use `data=df` for dataset .
Only return the plotting code.  don't use %matplotlib inline  and %config InlineBackend.figure_format = 'retina'
"""

    def run(self):
        st.set_page_config(page_title="📊 HR Visualizer", page_icon="📈")
        st.title("📊 Text to Data Visualization Tool")

        user_input = st.text_input("📝 Describe the chart you want to generate:")

        if user_input:
            with st.spinner("🧠 codellama is thinking..."):
                prompt = self.make_prompt(user_input)
                response = ollama.chat(
                    model="codellama",
                    messages=[{"role": "user", "content": prompt}]
                )
                generated_code = response['message']['content'].strip()

                # Clean markdown/codeblock syntax
                code_str = re.sub(r"^(```(python)?|```)$", "", generated_code, flags=re.MULTILINE).strip()

                # Show the generated code
                st.text_area("🧾 Generated Python Code:", value=code_str, height=150)

                # Execute and show the plot
                exec_env = {
                    "plt": plt,
                    "sns": sns,
                    "pd": pd,
                    "df": self.df
                }

                try:
                    exec(code_str, exec_env)
                    fig = plt.gcf()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Error executing the generated code:\n\n{e}")

# Instantiate and run the app
if __name__ == "__main__":
    app = HRVisualizerApp()
    app.run()
