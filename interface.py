import streamlit as st
import pandas as pd 
import random
import os
from groq import Groq
import plotly.graph_objects as go

Dataset = pd.read_csv('Submission\streamlit_table.csv.csv')


st.set_page_config(layout="wide")

# Define custom CSS
css = """
<style>
    /* Background color of the main content area */
    .stApp {
        background: linear-gradient(to right, #ADD8E6, #FFFFFF); /* Light blue to white gradient */
    }

    /* Sidebar color */
    .css-1d391kg {
        background: linear-gradient(to bottom, #ADD8E6, #FFFFFF); /* Light blue to white gradient */
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Display the main title
st.markdown(
    """
    <h1 style='font-size: 80px; text-align: center;'>Hiring with Satya 🔎</h1>
    """, 
    unsafe_allow_html=True
)

# Display the candidate database title
st.markdown(
    """
    <hr>
    <h1>Candidate Database</h1>
    """, 
    unsafe_allow_html=True
)
st.markdown(f"<hr>", unsafe_allow_html=True)
# Initialize session state variables
if 'view_profile' not in st.session_state:
    st.session_state.view_profile = False
if 'current_profile_id' not in st.session_state:
    st.session_state.current_profile_id = None
if 'offset' not in st.session_state:
    st.session_state.offset = 0
if 'limit' not in st.session_state:
    st.session_state.limit = 10  # Number of candidates to display at a time
if 'displayed_candidates' not in st.session_state:
    st.session_state.displayed_candidates = []  # Store the IDs of candidates to display

if 'active_tags' not in st.session_state:
    st.session_state.active_tags = []

# Define available tags
available_tags = ['Accounting', 'administrative', 'budgets', 'documentation', 'financial', 
                  'coaching', 'Excel', 'hardware', 'delivery', 'banking', 'inventory']

       
st.subheader("Skill Filter")
selected_tags = st.multiselect("Select Active Tags", available_tags, default=st.session_state.active_tags)

st.session_state.active_tags = selected_tags

if not st.session_state.active_tags:
    df_filtered = Dataset # No filter applied
    st.write("No filter is active. Showing all results.")
else:
    matching_indices = []

    # Iterate through the DataFrame
    for index, row in Dataset.iterrows():
        skills = row['Skills']
        
        # Check if skills is a list
        if isinstance(skills, list):
            # Use a nested for loop to check string matching
            for skill in skills:
                # Check if the skill matches any of the active tags
                if any(active_tag in skill for active_tag in st.session_state.active_tags):
                    matching_indices.append(index)
                    break  # Stop checking further skills for this row
    
    df_filtered = Dataset.loc[matching_indices]             
# Check if the DataFrame is empty
if Dataset.empty:
    st.markdown("<h3 style='font-size: 30px;'>No candidates available.</h3>", unsafe_allow_html=True)
else:
    # Check if we are in profile view or main list view
    if not st.session_state.view_profile:
        Dataset_sorted = df_filtered.sort_values(by='Overall_Score', ascending=False)

        # Calculate the total number of candidates to display
        total_candidates_to_display = st.session_state.offset + st.session_state.limit
        total_candidates = len(Dataset_sorted)

        # Update the displayed candidates list if necessary
        if total_candidates_to_display > len(st.session_state.displayed_candidates):
            additional_candidates = Dataset_sorted.index[st.session_state.offset:total_candidates_to_display].tolist()
            st.session_state.displayed_candidates.extend(additional_candidates)

       
        st.markdown(f"<hr> ", unsafe_allow_html=True)
        col1, col2, col3,col4,col5 = st.columns([1,1,1, 1, 1])
        with col1:
            st.markdown(f"<p style='font-size: 30px;'><b>ID</b></p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='font-size: 30px;'><b>Education</b></p>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<p style='font-size: 30px;'><b>Experience</b></p>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<p style='font-size: 30px;'><b>CV Score</b></p>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<p style='font-size: 30px;'><b>View Profile</b></p> ", unsafe_allow_html=True)
        st.markdown(f"<hr> ", unsafe_allow_html=True)
        for i in st.session_state.displayed_candidates:
                        # Create three columns: ID, CV Score, View Profile button
                    

            col1, col2, col3,col4,col5 = st.columns([1,1,1, 1, 1])  # Adjust the column width ratio as needed

            with col1:
                # Display the ID
                st.markdown(f"<p style='font-size: 30px;'>{Dataset['ID'][i]}</p>", unsafe_allow_html=True)
            with col2:
                # Display the ID
                st.markdown(f"<p style='font-size: 30px;'>{Dataset['Education_Level'][i]}</p>", unsafe_allow_html=True)

            with col3:
                # Display the ID
                st.markdown(f"<p style='font-size: 30px;'>{Dataset['Years_of_Experience'][i]}</p>", unsafe_allow_html=True)


            with col4:
                # Generate and display a random CV score between 0 and 10
                cv_score = Dataset["Overall_Score"]
                st.markdown(f"<p style='font-size: 30px;'> <b> {Dataset['Overall_Score'][i]:.2f} </b></p>", unsafe_allow_html=True)

            with col5:
                # Add a button to view profile
                 if st.button(f"View Profile for ID {Dataset['ID'][i]}"):
                    st.session_state.view_profile = True
                    st.session_state.current_profile_id = Dataset['ID'][i]

        # Display "Show More" button if there are more candidates to show
        if total_candidates_to_display < total_candidates:
            if st.button("Show More"):
                st.session_state.offset += st.session_state.limit  # Increment the offset
                st.experimental_rerun()  # Rerun the app to display more candidates
        else:
            st.markdown("<p style='font-size: 30px;'>Showing all candidates.</p>", unsafe_allow_html=True)

    else:
        # Display profile view based on current_profile_id
       
            profile_id = st.session_state.current_profile_id
            client = Groq(
                api_key='key',
            )
            
            def generate_resume_summary(resume_text):
                    prompt = f"Here is a resume text:\n{resume_text}\nGenerate a short summary with strengths and weaknesses of the candidate pointwise  bold at appropiate words 3 short points."
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama3-8b-8192",
                    )
                    return chat_completion.choices[0].message.content.strip()

                # Your existing code...

                # Assuming this is within the profile view section
            client = Groq(
                api_key='key',
            )

            def sentiment_calculator(text_path):

                with open(text_path, 'r', encoding='Windows-1252') as f:
                    text = f.read()

                # Initialize the Groq client
                client = Groq(
                    api_key='Key',
                )

                # Create the completion request with system and user messages
                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            You are provided a piece of text that contains various claims, both normal and exaggerated. Your task is to:
                            1. Identify all the claims in the text.
                            2. For normal claims, assign a sentiment score between 0 and 1.
                            3. For exaggerated claims, assign a sentiment score between 0 and 1.
                            4. Calculate the total sum of sentiment scores for normal claims and exaggerated claims separately.
                            5. Output the result as an integer, calculated using the formula:
                                (Average of normal sentiment scores) - 0.1 * (Average of exaggerated sentiment scores)

                            DO NOT Provide any intermediate steps in the response.

                            The final output should ONLY be the result of this formula.
                            """
                        },
                        {
                            "role": "user",
                            "content": f"""The text is provided below:\n {text}
                            
                            DO NOT Provide any intermediate steps in the response.

                            The final output should ONLY be the result of the formula mentioned above.
                            """
                        }
                    ],
                    temperature=0.5,
                    max_tokens=4096,
                    top_p=1,
                    stream=False,
                    stop=None,
                )

                # Return the summary generated by the model
                return completion.choices[0].message.content

           
            
            import matplotlib.colors as mcolors
            if st.session_state.view_profile:
                    profile_id = st.session_state.current_profile_id
                    st.markdown(f"<h3 style='font-size: 30px;'>Displaying profile for ID: {profile_id}</h3>", unsafe_allow_html=True)
                    
                    # Example PDF link
                    pdf_link = f"C:/Users/subar/OneDrive/Desktop/8Fold/Final_Resumes/Resume_of_ID_{profile_id}.pdf"  
                    pdf_display_name = f"Resume of ID {profile_id}"
                    # Display the PDF link
                    st.markdown(f"[{pdf_display_name}]({pdf_link})", unsafe_allow_html=True)

                    # Read resume data again to get the specific candidate's text
                    text_data = pd.read_csv('resume_text.csv')
                    col1, col2 ,col3  = st.columns([2, 0.08 ,1.6])  # Adjust the column width ratio as needed
                    with col1:
                        if 'Text' in text_data.columns:
                            resume_text = text_data["Text"][profile_id]

                            # Generate and display the resume summary
                            try:
                                summary = generate_resume_summary(resume_text)
                                st.markdown(f"<hr><h3 style='font-size: 30px;'><b>Resume Summary</h3></b><hr><p>{summary}</p><hr>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error generating resume summary: {str(e)}")
                                    
                    
                            def create_circular_bar(score, label, color):
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=score,
                                    title={'text': label, 'font': {'size': 20, 'color': "black"}},  # Title in bold black
                                    gauge={
                                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                        'bar': {'color': color},
                                        'bgcolor': "lightblue",  # Set background color to light blue
                                        'borderwidth': 2,
                                        'bordercolor': "black",
                                        'steps': [
                                            {'range': [0, 50], 'color': 'lightgray'},
                                            {'range': [50, 100], 'color': 'lightblue'}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},  # Optional: Threshold line can be customized
                                            'thickness': 0.75,
                                            'value': 70  # Optional: Example threshold value
                                        }
                                    },
                                    number={'font': {'color': "black", 'size': 20}}  # Number displayed in black
                                ))
                                
                                fig.update_layout(autosize=False, width=200, height=230, margin=dict(l=20, r=20, t=50, b=50))
                                return fig
                    with col2:
                        pass
                    with col3:
                        # Example scores for testing
                        impact_score = Dataset["Quantify impact_score"][profile_id] * 100 # Example score
                        brevity_score = Dataset["Brevity_Score"][profile_id] *100  # Example score
                        leadership_score =  Dataset["Managerial_CV_Score.1"][profile_id]*100
                        sections_score = Dataset["Section_Score"][profile_id]*100  # Example score
                        credibility_score = Dataset["CreditScore"][profile_id]  # Example score
                             # Usage
                        import glob

                        # Create a path with a wildcard to match any file
                        file_path_pattern = f'Final_Recommendation_Letters(1)/Recommendation_Letters_of_ID_{profile_id}/Recommendation_From_ID_*.txt'

                        # Use glob to find files matching the pattern
                        matching_files = glob.glob(file_path_pattern)

                        # Check if any files were found
                        if matching_files:
                            # Choose the first file from the list
                            file_path = matching_files[0]
                        
                            normalized_sentiment = float(sentiment_calculator(file_path)) * 100
                        else:
                            print("No files found matching the pattern.")

                        def score_to_color(score):
                            # Normalize score to be between 0 and 1
                            normalized_score = score / 100.0
                            # Calculate color components
                            red = 1 - normalized_score  # Red decreases as score increases
                            green = normalized_score      # Green increases as score increases
                            return mcolors.to_hex((red, green, 0))  # RGB format

                       
                       # Create the first row with 3 columns
                       # Create the first row with 3 columns
                      # Create the first row with 2 columns
                        score_col1, score_col2 = st.columns(2)

                        with score_col1:
                            st.plotly_chart(create_circular_bar(impact_score, "Impact", score_to_color(impact_score)), use_container_width=True)

                        with score_col2:
                            st.plotly_chart(create_circular_bar(brevity_score, "Brevity", score_to_color(brevity_score)), use_container_width=True)

                        # Create the second row with 2 columns
                        score_col3, score_col4 = st.columns(2)

                        with score_col3:
                            st.plotly_chart(create_circular_bar(leadership_score, "Leadership", score_to_color(leadership_score)), use_container_width=True)

                        with score_col4:
                            st.plotly_chart(create_circular_bar(sections_score, "Sections", score_to_color(sections_score)), use_container_width=True)

                        # Create the third row with 2 columns
                        score_col5, score_col6 = st.columns(2)

                        with score_col5:
                            st.plotly_chart(create_circular_bar(credibility_score, "Credibility", score_to_color(credibility_score)), use_container_width=True)
                        try:
                            with score_col6:
                                st.plotly_chart(create_circular_bar(normalized_sentiment, "Recommendation Sentiment", score_to_color(normalized_sentiment)), use_container_width=True)
                        except:
                            pass


                    import streamlit as st
                    from groq import Groq
                    import random

                    from langchain.chains import ConversationChain, LLMChain
                    from langchain_core.prompts import (
                        ChatPromptTemplate,
                        HumanMessagePromptTemplate,
                        MessagesPlaceholder,
                    )
                    from langchain_core.messages import SystemMessage
                    from langchain.chains.conversation.memory import ConversationBufferWindowMemory
                    from langchain_groq import ChatGroq
                    from langchain.prompts import PromptTemplate

                    # Get Groq API key
                    groq_api_key = 'key'  # Replace 'your_api' with your actual API key
                    col1, col2  = st.columns([1, 1])
                    # The title and greeting message of the Streamlit application
                    with col1:
                            
                        st.title("Ask question about the candidate")
                        st.write("Hello! I'm your friendly Chatbot. I can help answer your questions about the candidate, provide information, or just chat. I'm also super fast! Let's start our conversation!")

                        # Add customization options to the sidebar
                        system_prompt = f"this is the information about the candidate with ID {resume_text}"
                        model = 'llama3-8b-8192'
                        conversational_memory_length = 5

                        memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
                        user_question =  st.text_input("Ask a question:")
                        
                        st.write("Press Enter to send the question to the chatbot.")

                        # session state variable
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        else:
                            for message in st.session_state.chat_history:
                                memory.save_context(
                                    {'input': message['human']},
                                    {'output': message['AI']}
                                )

                        # Initialize Groq Langchain chat object and conversation
                        groq_chat = ChatGroq(
                            groq_api_key=groq_api_key, 
                            model_name=model
                        )

                        # If the user has asked a question,

        # If the user has asked a question,
                        if user_question:

                            # Construct a chat prompt template using various components
                            prompt = ChatPromptTemplate.from_messages(
                                [
                                    SystemMessage(
                                        content=system_prompt
                                    ),  # This is the persistent system prompt that is always included at the start of the chat.

                                    MessagesPlaceholder(
                                        variable_name="chat_history"
                                    ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                                    HumanMessagePromptTemplate.from_template(
                                        "{human_input}"
                                    ),  # This template is where the user's current input will be injected into the prompt.
                                ]
                            )

                            # Create a conversation chain using the LangChain LLM (Language Learning Model)
                            conversation = LLMChain(
                                llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
                                prompt=prompt,  # The constructed prompt template.
                                verbose=True,   # Enables verbose output, which can be useful for debugging.
                                memory=memory,  # The conversational memory object that stores and manages the conversation history.
                            )
                            
                            # The chatbot's answer is generated by sending the full prompt to the Groq API.
                            response = conversation.predict(human_input=user_question)
                            message = {'human':user_question,'AI':response}
                            st.session_state.chat_history.append(message)
                            st.write("Chatbot:", response)





                  
                    
            else:
                st.error("The resume text is not available for this candidate.")
            if st.button("Back to Candidate List"):
                    st.session_state.view_profile = False  # Reset the view profile state
                    st.session_state.offset = 0  # Reset the offset for candidates
                    st.session_state.displayed_candidates = []  # Clear displayed candidates for a fresh start
                    st.experimental_rerun()  # Rerun the app to refresh the display

# Sidebar content
st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 80px;'>Innov8 2.0</h1>
        <div style='display: flex; align-items: center; justify-content: center;'>
            <p style='font-size: 60px; margin: 0;'>eightfold.ai</p>
        </div>
    </div>
    <hr>
    <p style='font-size: 20px;' ,style='text-align: center;'>Team Name:<b> ModuleNotFound </b><br><br> Rwik Dey <br> Nisarg Bhavsar <br> Sachish Singla <br> Divyansh Sharma <br> Subarno Maji</p>
    """, 
    unsafe_allow_html=True
)


st.sidebar.markdown("</div>", unsafe_allow_html=True)