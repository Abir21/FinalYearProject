import streamlit as st 
from multipage import MultiApp

#calling the files from the apps folder
from apps import (
    stat,
    accept,
    tryAccept
    )

apps = MultiApp()

#This application is used to demonstrate the untapped potential within the FDP
#communitites globally and the potentially positive effect from harnessing it,
#upon the consistant UNHCR funding gaps which is directly tied to the livelihood of the FDPs world wide. 

st.markdown(
    """The Dummy UNHCR FDP (Skills Predictive Evaluation) Prediction Application
     
    """
)

#Adding the applications

apps.add_app("Synthetic FDP re-employement data", stat.app)
apps.add_app("FDP re-employement Scheme Selection", accept.app)
apps.add_app("Predictive Contribution Ranage", tryAccept.app)

#The main application page
apps.run()