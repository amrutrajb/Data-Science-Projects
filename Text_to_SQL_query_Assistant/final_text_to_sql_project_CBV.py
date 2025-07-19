import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.llms import Ollama
import ast, re
from io import StringIO
from contextlib import redirect_stdout


class SQLTextAssistant:
    def __init__(self):
        self.excel_path = r'D:\Downloads\InsightFlow AI\venv\Scripts\programs\HR_data_15.csv'
        self.db_filename = "test.db"
        self.table_name = "my_table"

        self.df = pd.read_csv(self.excel_path)
        self.engine = create_engine(f"sqlite:///{self.db_filename}")
        #with self.engine.begin() as conn:
        self.df.to_sql(self.table_name, con=self.engine, if_exists='replace', index=False)

        print(f"✅ Saved table '{self.table_name}' to SQLite database '{self.db_filename}'")

        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_filename}", sample_rows_in_table_info=3)

        # ✅ Replace OpenAI LLM with Ollama using Code LLaMA
        self.llm = Ollama(model="codellama:latest", temperature=0.2)

        # Define embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        self.few_shot_data = self._load_few_shots()




        self.vectorstore = Chroma.from_texts(
            texts=[
                f"{ex.get('Question', '')} {ex.get('SQLQuery', '')} {ex.get('Answer', '')}"
                for ex in self.few_shot_data
            ],
            embedding=self.embeddings,
            metadatas=self.few_shot_data
)

        self.example_selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore,
            k=1
        )

        self.example_prompt = PromptTemplate(
            input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
            template="Question: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
        )

        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=self.example_prompt,
            prefix=_mysql_prompt,
            suffix=PROMPT_SUFFIX,
            input_variables=["input", "table_info", "tok_k"]
        )

        self.chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.db,
            prompt=self.few_shot_prompt,
            verbose=True
        )

    def _load_few_shots(self):
        raw_examples = [
            {
                "Question": "what is the average headcount",
                "SQLQuery": 'SELECT AVG("headcount") FROM my_table',
                "SQLResult": "[(1106.0358974358974,)]",
                "Answer": "The average headcount is approximately 1106.04."
            },
            {
                "Question": "what is the total headcount",
                "SQLQuery": 'SELECT SUM("headcount") FROM my_table',
                "SQLResult": "[(431354,)]",
                "Answer": "The total headcount is 431354."
            },
            {
                "Question": "for year 2022 compare financial quater by quater comparision for Exit_count_involuntary column",
                "SQLQuery": """SELECT "Financial_Quarter", SUM("Exit_count_involuntary") as Total_Exit_Count_Involuntary
    FROM my_table
    WHERE "Year" = 2022
    GROUP BY "Financial_Quarter"
    ORDER BY "Financial_Quarter" ASC
    LIMIT 5;""",
                "SQLResult": "[('Q1', 497.0), ('Q2', 415.0), ('Q3', 169.0), ('Q4', 545.0)]",
                "Answer": "In 2022, the total involuntary exit count per quarter was: Q1 - 497, Q2 - 415, Q3 - 169, Q4 - 545."
            },
            {
                "Question": "Compare the average headcount for each division across quarters in 2022 and 2023. Which division showed the highest growth rate?",
                "SQLQuery": """WITH avg_headcount_per_year AS (
    SELECT division, "Year" AS year, AVG(headcount) AS avg_headcount
    FROM my_table
    WHERE "Year" IN (2022, 2023)
    GROUP BY division, "Year"
    ),
    growth_rate AS (
    SELECT a.division, a.avg_headcount AS avg_2022, b.avg_headcount AS avg_2023,
    ROUND(((b.avg_headcount - a.avg_headcount) / a.avg_headcount) * 100, 2) AS growth_rate_percent
    FROM avg_headcount_per_year a
    JOIN avg_headcount_per_year b
    ON a.division = b.division AND a.year = 2022 AND b.year = 2023
    )
    SELECT division, avg_2022, avg_2023, growth_rate_percent
    FROM growth_rate
    ORDER BY growth_rate_percent DESC
    LIMIT 1;""",
                "SQLResult": "[('Division_1', 3174.409090909091, 3459.125, 8.97)]",
                "Answer": "Division_1 had the highest growth rate in average headcount with 8.97% increase from 2022 to 2023."
            }
        ]

        # ✅ Auto-fill missing 'Answer' key if needed
        for example in raw_examples:
            example.setdefault("Answer", "")  # ensures 'Answer' key always exists

        return raw_examples


    def query(self, user_input: str):
        try:
            f = StringIO()
            with redirect_stdout(f):
                # Use proper key expected by SQLDatabaseChain
                _ = self.chain.invoke({"question": user_input})
            logs = f.getvalue()

            # Extract from logs since Ollama doesn't return structured dicts
            sql_query = re.search(r"SQLQuery:\s*(.*?)\s*SQLResult:", logs, re.DOTALL)
            sql_query = sql_query.group(1).strip() if sql_query else ""

            sql_result = re.search(r"SQLResult:\s*(.*?)\s*Answer:", logs, re.DOTALL)
            sql_result = sql_result.group(1).strip() if sql_result else ""

            answer = re.search(r"Answer:\s*(.*?)\s*> Finished chain", logs, re.DOTALL)
            answer = answer.group(1).strip() if answer else ""

            # Clean ANSI codes if any
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            sql_query = ansi_escape.sub('', sql_query)
            sql_result = ansi_escape.sub('', sql_result)
            answer = ansi_escape.sub('', answer)
            logs = ansi_escape.sub('', logs)

            # Extract column names for DataFrame
            column_list = []
            match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_query, re.IGNORECASE | re.DOTALL)
            if match:
                columns_str = match.group(1)
                raw_columns = [col.strip() for col in columns_str.split(",")]
                for col in raw_columns:
                    alias_match = re.search(r"AS\s+`?(\w+)`?", col, re.IGNORECASE)
                    if alias_match:
                        column_list.append(alias_match.group(1))
                    else:
                        clean_col = col.replace("`", "").split(".")[-1]
                        column_list.append(clean_col)

            # Convert result string to DataFrame
            try:
                parsed_result = ast.literal_eval(sql_result)
                df = pd.DataFrame(parsed_result, columns=column_list) if parsed_result else None
            except Exception as e:
                df = None
                st.error(f"❌ Failed to parse SQL result: {e}")

            return sql_query, answer, df, logs

        except Exception as e:
            st.error(f"❌ Error running query: {e}")
            return "", "", None, ""






# ─────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SQL Assistant", layout="wide")
st.markdown("## 🧠 Text to SQL Query Assistant")

assistant = SQLTextAssistant()

user_input = st.text_input("🔍 Enter your question:")

if st.button("🚀 Run Query") and user_input:
    with st.spinner("Processing your query..."):
        sql_query, answer, df, logs = assistant.query(user_input)

    st.write("**🧠 Generated SQL Query**")
    st.code(sql_query or "No SQL query found.", language="sql")

    st.write("**📝 Final Answer**")
    st.success(answer or "No answer found.")

    st.write("**📊 SQL Result Table**")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No tabular data to display.")

    with st.expander("🪵 Show Full Logs"):
        st.code(logs)

    st.caption("💡 Tip: Make sure your query is clear and refers to data available in your database.")
