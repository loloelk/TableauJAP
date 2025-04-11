# components/overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main_dashboard_page():
    """Main overview dashboard with key metrics"""
    # Create a layout with title on left and patient selection on right
    col_title, col_select = st.columns([2, 1])
    
    with col_title:
        st.header("Vue d'Ensemble - Premier Épisode Psychotique")
    
    with col_select:
        if hasattr(st.session_state, 'final_data') and not st.session_state.final_data.empty:
            # Get patient IDs and sort them
            all_patient_ids = sorted(st.session_state.final_data['ID'].unique().tolist())
            
            if all_patient_ids:
                # Create a horizontal layout for selection and button
                sel_col, btn_col = st.columns([3, 1])
                
                with sel_col:
                    selected_patient = st.selectbox(
                        "Sélectionner un patient:",
                        all_patient_ids,
                        index=0 if st.session_state.selected_patient_id not in all_patient_ids else all_patient_ids.index(st.session_state.selected_patient_id),
                        key="overview_patient_selector",
                        label_visibility="collapsed"  # Hide the label for cleaner layout
                    )
                
                with btn_col:
                    if st.button("Voir détails", type="primary", key="view_details_btn"):
                        st.session_state.selected_patient_id = selected_patient
                        st.session_state.sidebar_selection = "Tableau de Bord du Patient"
                        st.rerun()
    
    # Add a divider
    st.markdown("---")
    
    # Display error if no data
    if not hasattr(st.session_state, 'final_data') or st.session_state.final_data.empty:
        st.error("Aucune donnée patient chargée.")
        return
    
    # Top metrics in three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Count total patients
        total_patients = len(st.session_state.final_data)
        st.metric("Nombre Total de Patients", total_patients)
    
    with col2:
        # Calculate average PANSS positive improvement
        panss_df = st.session_state.final_data[
            st.session_state.final_data['panss_pos_bl'].notna() & 
            st.session_state.final_data['panss_pos_fu'].notna()
        ]
        
        if not panss_df.empty:
            improvement = panss_df['panss_pos_bl'] - panss_df['panss_pos_fu']
            avg_improvement = improvement.mean()
            st.metric("Amélioration Symptômes Positifs", f"{avg_improvement:.1f} points")
        else:
            st.metric("Amélioration Symptômes Positifs", "N/A")
    
    with col3:
        # Calculate response rate (>= 30% improvement in positive symptoms)
        if not panss_df.empty:
            percent_improvement = (improvement / panss_df['panss_pos_bl']) * 100
            response_rate = (percent_improvement >= 30).mean() * 100
            st.metric("Taux de Réponse (≥30%)", f"{response_rate:.1f}%")
        else:
            st.metric("Taux de Réponse", "N/A")
    
    # Create tabs for different overview sections
    tab1, tab2, tab3 = st.tabs([
        "📊 Distribution", 
        "📈 Réponse Clinique", 
        "🏥 Hospitalisations"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Antipsychotics distribution
            st.subheader("Distribution des Antipsychotiques")
            
            if 'medications' in st.session_state.final_data.columns:
                # Extract primary medication from the medications column
                med_data = st.session_state.final_data['medications'].str.split(';').str[0].str.split().str[0]
                med_counts = med_data.value_counts().reset_index()
                med_counts.columns = ['Antipsychotique', 'Nombre de Patients']
                
                # Create a pie chart
                fig = px.pie(
                    med_counts, 
                    values='Nombre de Patients',
                    names='Antipsychotique',
                    title="Répartition des Patients par Antipsychotique"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("La colonne 'medications' n'existe pas dans les données.")
        
        with col2:
            # Age distribution
            st.subheader("Distribution des Âges")
            
            if 'age' in st.session_state.final_data.columns:
                fig_age = px.histogram(
                    st.session_state.final_data,
                    x='age',
                    nbins=10,
                    title="Distribution des Âges",
                    labels={'age': 'Âge', 'count': 'Nombre de Patients'},
                    color_discrete_sequence=[st.session_state.PASTEL_COLORS[2]]
                )
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.warning("La colonne 'age' n'existe pas dans les données.")
    
    with tab2:
        st.subheader("Évolution des Scores PANSS")
        
        if 'panss_pos_bl' in st.session_state.final_data.columns and 'panss_pos_fu' in st.session_state.final_data.columns:
            # Create a waterfall chart to show overall improvement
            panss_scores = st.session_state.final_data[['ID', 'panss_pos_bl', 'panss_pos_fu']].dropna()
            
            if not panss_scores.empty:
                panss_scores['improvement'] = panss_scores['panss_pos_bl'] - panss_scores['panss_pos_fu']
                panss_scores['improvement_pct'] = (panss_scores['improvement'] / panss_scores['panss_pos_bl'] * 100).round(1)
                panss_scores = panss_scores.sort_values('improvement_pct', ascending=False)
                
                # Create bar chart
                fig_improvement = px.bar(
                    panss_scores,
                    x='ID',
                    y='improvement_pct',
                    title="Pourcentage d'amélioration des Symptômes Positifs par patient",
                    labels={'improvement_pct': "Amélioration (%)", 'ID': "Patient ID"},
                    color='improvement_pct',
                    color_continuous_scale='Blues'
                )
                
                # Update layout for better display
                fig_improvement.update_layout(
                    xaxis={'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(fig_improvement, use_container_width=True)
                
                # Add threshold lines for response and remission
                panss_scores_sorted = panss_scores.sort_values('ID')
                
                fig_before_after = go.Figure()
                fig_before_after.add_trace(go.Scatter(
                    x=panss_scores_sorted['ID'],
                    y=panss_scores_sorted['panss_pos_bl'],
                    mode='lines+markers',
                    name='Baseline',
                    line=dict(color=st.session_state.PASTEL_COLORS[0], width=2)
                ))
                fig_before_after.add_trace(go.Scatter(
                    x=panss_scores_sorted['ID'],
                    y=panss_scores_sorted['panss_pos_fu'],
                    mode='lines+markers',
                    name='Jour 30',
                    line=dict(color=st.session_state.PASTEL_COLORS[1], width=2)
                ))
                
                # Add threshold lines
                fig_before_after.add_shape(
                    type="line", line=dict(dash='dash', color='green', width=2),
                    x0=0, x1=1, xref="paper",
                    y0=15, y1=15, yref="y"
                )
                fig_before_after.add_annotation(
                    xref="paper", yref="y",
                    x=0.01, y=15,
                    text="Seuil d'amélioration clinique",
                    showarrow=False,
                    font=dict(color="green")
                )
                
                fig_before_after.update_layout(
                    title="Scores PANSS (sympt. positifs) avant et après traitement",
                    xaxis_title="Patient ID",
                    yaxis_title="Score PANSS (positif)"
                )
                
                st.plotly_chart(fig_before_after, use_container_width=True)
            else:
                st.warning("Données PANSS insuffisantes pour l'analyse.")
        else:
            st.warning("Les colonnes PANSS n'existent pas dans les données.")
    
    with tab3:
        st.subheader("Hospitalisations et Visites aux Urgences")
        
        # Add metrics for hospitalization statistics
        hosp_cols = st.columns(2)
        with hosp_cols[0]:
            if 'hospitalizations_past_year' in st.session_state.final_data.columns:
                avg_hosp = st.session_state.final_data['hospitalizations_past_year'].mean()
                st.metric("Moyenne d'Hospitalisations (12 mois)", f"{avg_hosp:.1f}")
            else:
                st.metric("Moyenne d'Hospitalisations", "N/A")
        
        with hosp_cols[1]:
            if 'er_visits_past_year' in st.session_state.final_data.columns:
                avg_er = st.session_state.final_data['er_visits_past_year'].mean()
                st.metric("Moyenne Visites Urgences (12 mois)", f"{avg_er:.1f}")
            else:
                st.metric("Moyenne Visites Urgences", "N/A")
        
        # Create hospitalization visualization if data is available
        if 'hospitalizations_past_year' in st.session_state.final_data.columns:
            hosp_data = st.session_state.final_data[['ID', 'hospitalizations_past_year', 'er_visits_past_year']].dropna()
            
            if not hosp_data.empty:
                hosp_data = hosp_data.sort_values('hospitalizations_past_year', ascending=False)
                
                fig_hosp = px.bar(
                    hosp_data,
                    x='ID',
                    y=['hospitalizations_past_year', 'er_visits_past_year'],
                    title="Hospitalisations et Visites aux Urgences par Patient",
                    labels={
                        'value': "Nombre",
                        'ID': "Patient ID",
                        'variable': "Type"
                    },
                    barmode='group'
                )
                
                # Rename the legend items
                fig_hosp.update_layout(
                    legend_title="Type de Visite",
                    xaxis={'categoryorder': 'total descending'}
                )
                
                # Update names in the legend
                newnames = {'hospitalizations_past_year': 'Hospitalisations', 
                           'er_visits_past_year': 'Visites Urgences'}
                fig_hosp.for_each_trace(lambda t: t.update(name = newnames[t.name]))
                
                st.plotly_chart(fig_hosp, use_container_width=True)
            else:
                st.warning("Données d'hospitalisation insuffisantes pour l'analyse.")
        else:
            st.warning("Les colonnes d'hospitalisation n'existent pas dans les données.")