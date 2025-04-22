import streamlit as st  
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import os 
import plotly.graph_objects as go
import plotly.express as px 
from streamlit_option_menu import option_menu
import time

def all_estimated_mh(path, req):
    estimated_mh = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
    df_estimated = pd.DataFrame(estimated_mh)
    df_estimated = df_estimated[df_estimated["header"].isin(req)]
    print(df_estimated)
    task_estimated = sorted(list(set(df_estimated['task name'].to_list())))
    estimated_mh_array = np.zeros(len(task_estimated))
    for i in range(len(task_estimated)):
        estimated_mh_array[i] = df_estimated[df_estimated['task name']==task_estimated[i]]['Estimated'].sum()
    return estimated_mh_array, df_estimated

def prepend_number_if_contains(value, substring, number):  
    if pd.isna(value):  # Handle NaN values if they exist  
        return value  
    if substring in value:  
        return number + value  
    return value  

def substitute_task_num(path):
    MH_summary = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
    df_MH = pd.DataFrame(MH_summary)
    df_MH['task name'] = df_MH['task name'].apply(lambda x: prepend_number_if_contains(x, 'Software Requirement', '1. '))
    df_MH['task name'] = df_MH['task name'].apply(lambda x: prepend_number_if_contains(x, 'Detail Design', '2. '))
    df_MH['task name'] = df_MH['task name'].apply(lambda x: prepend_number_if_contains(x, 'Implementation', '3. '))
    df_MH['task name'] = df_MH['task name'].apply(lambda x: prepend_number_if_contains(x, 'Software Test', '4. '))
    df_MH['task name'] = df_MH['task name'].apply(lambda x: prepend_number_if_contains(x, 'Documentation', '5. '))
    return df_MH

def MH_for_each_req(df_MH,req):
    grouped_data = df_MH[df_MH["header"]==req].groupby(['header',"task name",'category','username'])
    df_MH_23_429 = grouped_data['timeSpent'].sum().reset_index().sort_values(by=["task name",'category','timeSpent'], ascending=[True,False,False])
    return df_MH_23_429

def MH_for_all_req(df_MH, req_list):
    df_MH = df_MH[df_MH['header'].isin(req_list)]
    df_MH.reset_index(drop=True, inplace=True)
    df_MH['task name'] = df_MH['task name'].fillna('')
    task_list = ['Software Requirement', 'Detail Design', 'Implementation', 'Software Test','Documentation']
    task_MH = []
    phase_list = []
    for task in task_list :
        MH = df_MH[df_MH["task name"].str.contains(task)]['timeSpent'].sum()
        phase_list.append(df_MH.loc[0,"phase"])
        task_MH.append(float(MH))
    df_task_MH = pd.DataFrame({"phase":phase_list,"task name":task_list,"timeSpent":task_MH})
    return df_task_MH

def req_tolist(path):
    MH_summary = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
    df_MH = pd.DataFrame(MH_summary)
    filter = ~((df_MH['header'].str.contains("Project Management"))|(df_MH['header'].str.contains("Project Support"))|(df_MH['header'].str.contains("Temp")))
    df_MH = df_MH[filter]
    req_list = list(set(df_MH['header'].to_list()))
    return req_list

class ManHoursAnalysis:
    def __init__(self, req_df, estimated_df, req_num):  
        self.req_df = req_df  
        self.estimated_df = estimated_df
        self.req_num = req_num
        self.task_name = None  
        self.username = None  
        self.len_username = None  
        self.len_task_name = None  
        self.actual_mh_array = None  
        self.estimated_mh_array = None  
  
    def calculate_mh_array(self):
        print(f"Req. {self.req_num}")
        estimated_task_list_raw = sorted(list(set(self.estimated_df[self.estimated_df['header']==self.req_num]["task name"].to_list())))
        estimated_task_list = [s[:-1] if s.endswith(" ") else s for s in estimated_task_list_raw]
        #estimatd_task_list = [f'1. Software Requirement ({self.req_num})', f'2. Detail Design ({self.req_num})', f'3. Implementation ({self.req_num})', f'4. Software Test ({self.req_num})']
        #self.task_name = sorted(list(set(self.req_df["task name"].tolist())))  
        self.username = sorted(list(set(self.req_df["username"].tolist())))  
        self.task_name = [x + f" ({self.req_num})" for x in estimated_task_list]
        
        print(f"List of estimated tasks -> {estimated_task_list}")
        print(f"List of tasks -> {self.task_name}")  
        print(f"List of users -> {self.username}")  

        self.len_task_name = len(self.task_name)  
        self.len_username = len(self.username)  

        self.estimated_mh_array = self.estimated_df[self.estimated_df['header'] == self.req_num]['Estimated'].astype(float).round(2).to_list()
        print(f"Estimated MH Array \n {self.estimated_mh_array}")
  
        # Initialize the actual man-hours array with zeros  
        self.actual_mh_array = np.zeros((self.len_username, self.len_task_name))   
        
        for i in range(self.len_username):  
            for j in range(self.len_task_name):  
                time_spent = self.req_df[(self.req_df["username"] == self.username[i]) &   
                                         (self.req_df["task name"] == self.task_name[j])]['timeSpent'].sum()  
                self.actual_mh_array[i, j] = round(time_spent,2)

        print(f"Actual MH Array \n {self.actual_mh_array}")  
        print("--------------------END-------------------")

        return self.task_name,self.username,self.len_username,self.len_task_name,self.actual_mh_array,self.estimated_mh_array
    
    def plot_stacked_bar(self):
        if self.task_name is None or self.username is None or self.actual_mh_array is None or self.estimated_mh_array is None:  
            raise ValueError("Please run calculate_mh_array() before plotting.")
        #plot
        fig = go.Figure()

        #Actual
        total_actuals = np.sum(self.actual_mh_array, axis=0)
        for n in range(self.len_username):
            fig.add_trace(go.Bar(
                x = self.task_name,
                y = self.actual_mh_array[n,:],
                name = self.username[n],
                # text = self.actual_mh_array[n, :],
                # textposition='outside'
            ))
        
        fig.add_trace(go.Scatter(
            x = self.task_name,
            y = total_actuals,
            mode='text',
            text=total_actuals,
            textposition='top center',
            showlegend=False
        ))
        

        #Estimated
        fig.add_trace(go.Scatter(
            x = self.task_name,
            y = self.estimated_mh_array,
            mode='lines+markers+text',
            name = "Estimated",
            text = self.estimated_mh_array,
            line=dict(color='rgba(250, 156, 28, 1)', width=5),   
            marker=dict(symbol='circle', size=8),
            textposition= 'top center'  
        ))

        #merged chart
        fig.update_layout(
            title=f"Actual MH vs. Estimated MH of Req. {self.req_num}",  
            xaxis=dict(title="Task"),  
            yaxis=dict(title="MH"),  
            barmode="stack",
            width=1000,
            height=500
        )

        return fig

class mh_summary:
    def __init__(self,df_all_mh, plan_pg, actual_pg):
        self.df_all_mh = df_all_mh
        self.plan_pg = plan_pg
        self.actual_pg = actual_pg

    def schedule_delay(self):
        delay =  self.plan_pg - self.actual_pg
        return delay

    def mh_deviation(self):
        if self.df_all_mh is not None :
            actual_mh = self.df_all_mh["timeSpent"].sum()
            estimated_mh = self.df_all_mh["Estimated"].sum()
            mh_percent = (actual_mh/estimated_mh)*100
            deviation = mh_percent - self.actual_pg
        return deviation

def highlight_cells(cell, est_col, act_col):  
    if cell[act_col] > cell[est_col]:  
        return [''] * cell.index.get_loc(act_col) + ['background-color: yellow'] + [''] * (len(cell) - cell.index.get_loc(act_col) - 1)  
    else:  
        return [''] * len(cell)  

def mh_summary_plot(x,y1,y2,y3,n1,n2,n3):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x = x,
        y = y2,
        name = n2,
        text=y2,
        textposition='outside',
        opacity = 0.5,
        offsetgroup=0,
        marker=dict(color='blue')
    ))

    fig.add_trace(go.Bar(
        x = x,
        y = -y3,
        name = n3,
        text=y3,
        textposition='outside',
        opacity = 0.5,
        offsetgroup=0,
        marker=dict(color='red')
    ))

    fig.add_trace(go.Bar(
        x = x,
        y = y1,
        name = n1,
        text=y1,
        textposition='outside',
        opacity = 1,
        offsetgroup=0,
        marker=dict(color='yellow')
    ))

    fig.update_layout(title=f"MH Summary",  
        xaxis=dict(title="Requirement"),  
        yaxis=dict(title="MH"),  
        barmode="group",
        width=1000,
        height=500
    )
    return fig

def mh_for_each_task(df):
    category = sorted(list(set(df["category"].to_list())),reverse=True)
    username = sorted(list(set(df["username"].to_list())))

    len_category = len(category)
    len_username = len(username)

    init_array = np.zeros((len_category, len_username))
    for i in range(len_category):
        for j in range(len_username):
            timespent = df[(df["username"]==username[j])&(df["category"]==category[i])]["timeSpent"].sum()
            init_array[i,j] = round(timespent,2)

    df_merge = pd.DataFrame(init_array, index=category, columns=username)
    df_merge["Total MH"] = df_merge.sum(axis=1)
    df_merge.loc["Total MH"] = df_merge.sum(axis=0)

    return df_merge, category, username

def req_col_tolist(path):
    df = pd.read_excel(path, engine="openpyxl")
    mh_df = pd.DataFrame(df)
    req_list = list(set(mh_df["header"].to_list()))
    return req_list

def each_person(path,req):
    df_mh = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
    df_mh = df_mh[df_mh["header"].isin(req)]
    group_mh = df_mh.groupby('username')
    df_each_person = group_mh["timeSpent"].sum().reset_index().sort_values("timeSpent", ascending=True, ignore_index=True)
    print(df_each_person)
    
    fig = px.bar(df_each_person, x='timeSpent', y='username', orientation='h',
                title='Overall MH for each people',  
                labels={'MH': "timeSpent", 'Team Member': 'username'}
    ) 
    
    fig.update_layout(  
        title_text='Overall MH for each people',  
        xaxis_title='MH',  # X-axis label  
        yaxis_title='Team Member',  # Y-axis label  
    ) 

    return df_each_person, fig
    
# Directory to save uploaded files  
UPLOAD_DIR = "uploaded_files"  
  
# Ensure the upload directory exists  
if not os.path.exists(UPLOAD_DIR):  
    os.makedirs(UPLOAD_DIR)  
  
def save_uploaded_file(uploaded_file):  
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)  
    with open(file_path, "wb") as f:  
        f.write(uploaded_file.getbuffer())  
    return file_path  
  
def get_saved_files():  
    return os.listdir(UPLOAD_DIR)  
  
def main():  
    # Set the title of the Streamlit app  
    st.title("Man-Hours Analysis")  
      
    # Add a sidebar for file upload and requirement selection  
    st.sidebar.header("Upload and Select Options")  
      
    # File uploader to upload the Excel files  
    uploaded_file_1 = st.sidebar.file_uploader("Choose an Actual MH file", type="xlsx", accept_multiple_files=True)  
    uploaded_file_2 = st.sidebar.file_uploader("Choose an Estimated MH file", type="xlsx", accept_multiple_files=True)  
      
    # Save uploaded files  
    if uploaded_file_1 is not None:  
        actual_path = save_uploaded_file(uploaded_file_1)   
    if uploaded_file_2 is not None:  
        estimated_path = save_uploaded_file(uploaded_file_2) 
  
    # Get saved files  
    saved_files = get_saved_files()  
      
    # Separate actual and estimated files  
    actual_files = [f for f in saved_files if "summary" in f.lower()]  
    estimated_files = [f for f in saved_files if "estimated" in f.lower()]  
  
    # Dropdown selector to select from previously uploaded files  
    if actual_files:  
        selected_actual_file_name = st.sidebar.selectbox("Select an Actual MH file from previous uploads", actual_files)  
        selected_actual_file = os.path.join(UPLOAD_DIR, selected_actual_file_name)    
      
    if estimated_files:  
        selected_estimated_file_name = st.sidebar.selectbox("Select an Estimated MH file from previous uploads", estimated_files)  
        selected_estimated_file = os.path.join(UPLOAD_DIR, selected_estimated_file_name)  
  
    # If both actual and estimated files are selected, perform further analysis  
    if actual_files and estimated_files: 
        df_MH = substitute_task_num(selected_actual_file)   
        headerbar = option_menu(None, ["Overall", "Requirement","Database"], 
        icons=['file-earmark-text-fill', 'clipboard-data','database-fill'], 
        menu_icon= "cast", default_index=0, orientation="horizontal")

        req_list = req_tolist( selected_actual_file)
        estimated_df = all_estimated_mh(selected_estimated_file,req_list)[1]
        df_mh_2 = substitute_task_num( selected_actual_file)
        for req in req_list:
            if req not in list(set(estimated_df["header"].to_list())):
                print("CANCEL",req,req_list)
                req_list.remove(req)
            else:
                None
        print(f"Requirement list {req_list}")

        if 'selected_option' not in st.session_state:  
            st.session_state.selected_option = None

        cancel_req = req_list
        st.sidebar.header("Fill Out Requirement")
        multi_side = st.sidebar.multiselect(  
            'Fill Out:',  cancel_req,
            default = st.session_state.selected_option
        )
        for e_req in multi_side:
            req_list.remove(e_req)

        each_p_fig = each_person(selected_actual_file,req_list)[1]

        if headerbar == "Overall":
            all_req_df = MH_for_all_req(df_MH, req_list)
            all_req_df.reset_index(drop=True, inplace=True)
            all_req_df["Estimated"] = all_estimated_mh(selected_estimated_file,req_list)[0]
            sum_actual = all_req_df["timeSpent"].sum()
            sum_estimated = all_req_df["Estimated"].sum()
            total_row = pd.DataFrame({col: [value] for col, value in zip(all_req_df.columns.to_list(), ["", "Total", sum_actual, sum_estimated])})
            print(total_row)
            new_all_req_df = pd.concat([all_req_df,total_row], ignore_index=True)
            all_req_style_df = new_all_req_df.style.apply(lambda row: highlight_cells(row,"Estimated","timeSpent"), axis=1)

            # Display the dataframe for the selected requirement  
            st.subheader("Man-Hours for All Requirement")
            st.write('')

            # start_d = st.date_input("Start Date")
            # end_d = st.date_input("End Date")
            # Initialize session state for number inputs  
            if 'plan_progress' not in st.session_state:  
                st.session_state.plan_progress = 0.00
            if 'actual_progress' not in st.session_state:  
                st.session_state.actual_progress = 0.00 

            # Number inputs for progress  
            st.session_state.plan_progress = st.number_input("Plan Progress", value=st.session_state.plan_progress, step=0.01, format="%.2f")  
            st.session_state.actual_progress = st.number_input("Actual Progress", value=st.session_state.actual_progress, step=0.01, format="%.2f")  
            
            col1,col2 = st.columns([0.07,1])
            with col2:
                st.write('Actual MH vs. Estmated MH Table')
                # Toggle button to show/hide dataframe  
            with col1:
                if st.button('📄'):
                    st.session_state.show_table = not st.session_state.get('show_table', True)
            # Display the dataframe based on the button state  
            if st.session_state.get('show_table', True):
                col_df1, col_df2 = st.columns([3,1])
                with col_df1:
                    st.dataframe(all_req_style_df)
                with col_df2:
                    mh_sum = mh_summary(all_req_df,st.session_state.plan_progress,st.session_state.actual_progress)
                    st.write(f'MH Deviation   : {mh_sum.mh_deviation():.2f}%')
                    st.write(f'Schedule Delay : {mh_sum.schedule_delay():.2f}%')
                    if st.button("Details"):  
                        st.session_state.page = "details"
            st.write('')
            
            if 'show_chart' not in st.session_state:  
                st.session_state.show_chart = True

            # Set up the Streamlit columns  
            t_col_1, t_col_2 = st.columns([0.07, 1])  
            
            with t_col_2:  
                st.write("Actual MH vs. Estimated MH Chart")  
            
            with t_col_1:  
                if st.button('📊'):  
                    st.session_state.show_chart = not st.session_state.show_chart
            
            if st.session_state.show_chart:  
                # Create a Plotly figure  
                fig = go.Figure()  
                
                # Actual MH  
                fig.add_trace(go.Bar(  
                    x=all_req_df["task name"].tolist(),  
                    y=all_req_df["timeSpent"].tolist(),  
                    name="Actual MH",  
                    text=all_req_df["timeSpent"].astype(float).round(2).tolist(),  
                    offsetgroup=0  
                ))  
                
                # Estimated MH  
                fig.add_trace(go.Scatter(  
                    x=all_req_df["task name"].tolist(),  
                    y=all_req_df["Estimated"].tolist(), 
                    mode='lines+markers+text', 
                    name="Estimated MH",  
                    text=all_req_df["Estimated"].astype(float).round(2).tolist(),  
                    line=dict(color='rgba(250, 156, 28, 1)', width=5),    
                    marker=dict(symbol='circle', size=8),
                    textposition= 'top center' 
                ))
                    
                fig.update_layout(  
                    title="Actual MH vs. Estimated MH",  
                    xaxis=dict(title="Task"),  
                    yaxis=dict(title="MH"),  
                    barmode="group",
                    width=1000,
                    height=500
                )  
                
                st.plotly_chart(fig,use_container_width=True)
        
        if headerbar == "Requirement":
            if 'sub_sidebar' not in st.session_state:  
                st.session_state.sub_sidebar = "Summary"

            st.sidebar.header("Requirement")

            options = ["Summary","Details"]
            sub_sidebar = st.sidebar.radio("Select", options, index=options.index(st.session_state.sub_sidebar))
            st.session_state.sub_sidebar = sub_sidebar

            if sub_sidebar == "Summary":
                st.subheader("Man-Hours Summary")
                st.write("")

                total_act = []
                total_est = []
                for each_req in req_list:
                    each_req_df = MH_for_each_req(df_mh_2, each_req)
                    mh_a = ManHoursAnalysis(each_req_df, estimated_df, each_req)
                    act_mh = mh_a.calculate_mh_array()[4]
                    act_mh_total = round(np.sum(act_mh),2)
                    print(f"Total Actual Amount {act_mh_total}")
                    est_mh = mh_a.calculate_mh_array()[5]
                    est_mh_total = round(np.sum(est_mh),2)
                    print(f"Total Estimated Amount {est_mh_total}")
                    total_act.append(act_mh_total)
                    total_est.append(est_mh_total)
                total_dif = np.array(total_est) - np.array(total_act)
                abs_total_dif = abs(total_dif)
                print(total_dif)
                print(abs_total_dif)

                #add metrics
                col_m1,col_m2,col_m3,col_m4,col_m5 = st.columns(5)

                st.markdown(  
                    """  
                    <style>  
                    .metric {  
                        /* border: 2px solid #c5d9ed;*/  /* Border color and thickness */  
                        border-radius: 5px; /* Rounded corners */  
                        padding: 10px; /* Padding inside the border */  
                        margin: 5px; /* Margin outside the border */  
                        background-color: #2e3338;
                    }  
                    .metric-label {  
                        font-size: 12px;  
                        font-weight: bold;
                        color: #d63638;  
                    }  
                    .metric-value {  
                        font-size: 20px;  
                        font-weight: bold;
                        color: #f6f7f7;  
                    }  
                    .metric-delta {  
                        font-size: 16px;  
                        color: #f6f7f7;  
                    }  
                    </style>  
                    """,  
                    unsafe_allow_html=True  
                )  

                # Create a function to generate metric HTML  
                def create_metric(label, value, delta):  
                    return f"""  
                    <div class="metric">  
                        <div class="metric-label">{label}</div>  
                        <div class="metric-value">{value}</div>  
                        <div class="metric-delta">{delta}</div>  
                    </div>  
                    """  
                                
                # Add metrics with custom CSS classes and border styling using parameters  
                col_m1.markdown(create_metric("Under Estimate MH" ,f"REQ. {req_list[np.where(total_dif==max(total_dif))[0][0]]}", f"{max(total_dif)} Hour"), unsafe_allow_html=True)  
                col_m2.markdown(create_metric("Over Estimate MH" ,f"REQ. {req_list[np.where(total_dif==min(total_dif))[0][0]]}", f"{min(total_dif)} Hour"), unsafe_allow_html=True) 
                col_m3.markdown(create_metric("Most Diffence MH" ,f"REQ. {req_list[np.where(abs_total_dif==max(abs_total_dif))[0][0]]}", f"{max(abs_total_dif)} Hour"), unsafe_allow_html=True)
                col_m4.markdown(create_metric("Most Usage MH", f"REQ. {req_list[total_act.index(max(total_act))]}", f"{max(total_act)} Hour"), unsafe_allow_html=True)
                col_m5.markdown(create_metric("Min Usage MH", f"REQ. {req_list[total_act.index(min(total_act))]}", f"{min(total_act)} Hour"), unsafe_allow_html=True)  

                #plot
                df_dif_mh = pd.DataFrame({"req_list":req_list, "total_dif":np.round(total_dif,2), "total_est":total_est,"total_act": np.array(total_act)})
                print(df_dif_mh)
                df_dif_mh = df_dif_mh.sort_values("total_dif", ascending=True, ignore_index=True)
                fig_dif = mh_summary_plot(df_dif_mh["req_list"],df_dif_mh["total_dif"],df_dif_mh["total_est"],df_dif_mh['total_act'],"Different MH","Estimated MH","Actual MH")
                st.plotly_chart(fig_dif, use_container_width=True)
                st.write('')
                st.plotly_chart(each_p_fig, use_container_width=True)

            if sub_sidebar == "Details" :
                st.subheader("Man-Hours for Each Requirement")
                st.write("")
                # col_2_1,col_2_2 = st.columns([5,100])
                # with col_2_1:
                #     st.write("List of Requirement ")
                
                if 'selected_req' not in st.session_state:  
                    st.session_state.selected_req = None

                st.session_state.selected_req = st.selectbox("Select a Requirement", req_list, index=req_list.index(st.session_state.selected_req) if st.session_state.selected_req else 0)  
                st.write('')  
                st.write('') 

                if st.session_state.selected_req:
                    req_df = MH_for_each_req(df_mh_2, st.session_state.selected_req)
                    req_df.reset_index(drop=True, inplace=True)
                    col2_1,col2_2 = st.columns([0.07,1])
                    with col2_2:
                        st.write(f"Data for Requirement: {st.session_state.selected_req}")
                    mh_analysis = ManHoursAnalysis(req_df, estimated_df, st.session_state.selected_req)
                    actual_mh_array = mh_analysis.calculate_mh_array()[4]
                    estimated_mh_array = mh_analysis.calculate_mh_array()[5]
                    task = mh_analysis.calculate_mh_array()[0]
                    username = mh_analysis.calculate_mh_array()[1]
                    actual_mh_array_t = actual_mh_array.transpose()
                    df_merge = pd.DataFrame(actual_mh_array_t, index=task, columns=username)
                    df_merge["Actual MH"] = df_merge.sum(axis=1)
                    df_merge["Estimated MH"] = estimated_mh_array
                    df_merge.loc["Grand"] = df_merge.sum(axis=0)
                    df_merge_style = df_merge.style.apply(lambda row: highlight_cells(row,"Estimated MH","Actual MH"), axis=1)  

                    with col2_1:
                        if st.button('📄'):
                            st.session_state.show_table = not st.session_state.get('show_table', True)
                    if st.session_state.get('show_table', True):
                        st.dataframe(df_merge_style)
                    st.write('')

                    if 'show_chart' not in st.session_state:  
                        st.session_state.show_chart = True
    
                    t_col_2_1, t_col_2_2 = st.columns([0.07, 1])
                    with t_col_2_2: 
                        st.write(f'Actual MH vs. Estimated MH of Req. {st.session_state.selected_req} Chart')
                    
                    with t_col_2_1:
                        if st.button('📊'):  
                            st.session_state.show_chart = not st.session_state.show_chart
                    if st.session_state.show_chart:
                        fig =  mh_analysis.plot_stacked_bar()
                        st.plotly_chart(fig, use_container_width=True)

                    st.write("")

                    st.write("--------------------------------------------------------------------------------------")

                    st.subheader("Man-Hours for Each Task")

                    if 'selected_task' not in st.session_state:  
                        st.session_state.selected_task = None


                    multi_select = st.sidebar.multiselect(  
                        'Select Task:',  task,
                        default = st.session_state.selected_task
                    )

                    all_1 = st.sidebar.checkbox("Select all", value=True)

                    if all_1:
                        multi_select = task

                    for task_sel in multi_select:
                        st.write(f"Data Table of {task_sel} task")
                        task_df,categry,name = mh_for_each_task(req_df[req_df["task name"] == task_sel])
                        #st.dataframe(req_df[req_df["task name"] == task_sel])
                        st.dataframe(task_df)
                        st.write("")

                        task_fig = go.Figure()
                        
                        for user in name:
                            task_fig.add_trace(go.Bar(
                                x = categry,
                                y = task_df[user],
                                name = user
                            ))

                        task_fig.update_layout(title=f"MH of each category for {task_sel} task chart ",  
                            xaxis=dict(title="Category"),  
                            yaxis=dict(title="MH"),  
                            barmode="stack",
                            width=1000,
                            height=500
                        )

                        st.plotly_chart(task_fig, use_container_width=True)
                        st.write("--------------------------------------------------------------------------------------")

        if headerbar == "Database":
            #st.subheader("Raw DATA")
            st.dataframe(df_MH.drop(columns=["Unnamed: 0","date"]))
                
    else:
        st.subheader("Please choose the estimated and actual in same phase")
if __name__ == "__main__":  
    main()  
