# 💡 Example Questions Section
st.header("💡 Example Questions")

# Category 1: Leaves & Attendance
st.subheader("🍃 Leaves & Attendance")
col1, col2, col3 = st.columns(3)

selected_question = None

with col1:
    if st.button("How many annual leave days are employees entitled to?", use_container_width=True):
        selected_question = "How many annual leave days are employees entitled to?"
    if st.button("What is the policy for sick leave?", use_container_width=True):
        selected_question = "What is the policy for sick leave?"

with col2:
    if st.button("What are the standard working hours?", use_container_width=True):
        selected_question = "What are the standard working hours?"
    if st.button("What is the maternity/paternity leave policy?", use_container_width=True):
        selected_question = "What is the maternity/paternity leave policy?"

with col3:
    if st.button("How do I request emergency leave?", use_container_width=True):
        selected_question = "How do I request emergency leave?"
    if st.button("What happens if I arrive late to work?", use_container_width=True):
        selected_question = "What happens if I arrive late to work?"

# Category 2: Work Environment & Perks
st.subheader("💻 Work Environment & Benefits")
col4, col5, col6 = st.columns(3)

with col4:
    if st.button("Can employees work remotely?", use_container_width=True):
        selected_question = "Can employees work remotely?"
    if st.button("What is the employee dress code?", use_container_width=True):
        selected_question = "What is the employee dress code?"

with col5:
    if st.button("What health insurance benefits are provided?", use_container_width=True):
        selected_question = "What health insurance benefits are provided?"
    if st.button("Is there a performance bonus policy?", use_container_width=True):
        selected_question = "Is there a performance bonus policy?"

with col6:
    if st.button("What is the policy for expense reimbursement?", use_container_width=True):
        selected_question = "What is the policy for expense reimbursement?"
    if st.button("What are the rules regarding notice period?", use_container_width=True):
        selected_question = "What are the rules regarding notice period?"

st.markdown("---")
