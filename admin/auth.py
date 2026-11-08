import hashlib
import hmac
import streamlit as st


def check_login():
    if st.session_state.get("admin_logged_in", False):
        return True

    st.title("Admin login")
    st.write("Log in om de databestanden te beheren.")

    with st.form("admin_login"):
        username = st.text_input("Gebruikersnaam")
        password = st.text_input("Wachtwoord", type="password")
        submitted = st.form_submit_button("Inloggen")

    if submitted:
        try:
            admin = st.secrets["admin"]
            expected_user = admin["username"]
            expected_hash = admin["password_hash"]

            password_hash = hashlib.sha256(password.encode()).hexdigest()

            if (
                hmac.compare_digest(username, expected_user)
                and hmac.compare_digest(password_hash, expected_hash)
            ):
                st.session_state["admin_logged_in"] = True
                st.rerun()
            else:
                st.error("Onjuiste gebruikersnaam of wachtwoord.")

        except Exception:
            st.error("Admin-instellingen ontbreken of zijn ongeldig.")

    return False


def logout():
    st.session_state["admin_logged_in"] = False
    st.rerun()