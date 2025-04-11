# components/side_effects.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Import the necessary functions from nurse_service.py
from services.nurse_service import get_side_effects_history, save_side_effect_report

def side_effect_page():
    """Page for tracking treatment side effects"""
    st.header("Suivi des Effets Secondaires")
    
    if not st.session_state.get("selected_patient_id"):
        st.warning("Aucun patient sélectionné.")
        return
    
    # Get patient side effect history from the nurse service
    # This is the key change - using the service function instead of direct CSV reading
    patient_side_effects = get_side_effects_history(st.session_state.selected_patient_id)
    
    # Display existing side effects if any
    if not patient_side_effects.empty:
        st.subheader("Effets secondaires signalés")
        
        # Display as a table with formatted column names
        display_df = patient_side_effects.copy()
        
        # Create mapping for column renaming
        rename_map = {
            'patient_id': 'ID Patient',
            'report_date': 'Date',
            'eps': 'Sympt. extrapyramidaux',
            'akathisia': 'Akathisie',
            'weight_gain': 'Prise de poids',
            'metabolic_changes': 'Chgt. métaboliques',
            'sedation': 'Sédation',
            'sexual_dysfunction': 'Dysf. sexuelle',
            'other_effects': 'Autre',
            'notes': 'Notes'
        }
        
        # Rename only the columns that exist in the DataFrame
        columns_to_rename = {col: rename_map[col] for col in rename_map if col in display_df.columns}
        display_df = display_df.rename(columns=columns_to_rename)
        
        # Select only display columns that exist
        display_columns = ['ID Patient', 'Date', 'Sympt. extrapyramidaux', 'Akathisie', 
                          'Prise de poids', 'Chgt. métaboliques', 'Sédation', 'Dysf. sexuelle', 'Autre', 'Notes']
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        # Display the dataframe with only the selected columns
        st.dataframe(display_df[display_columns])
        
        # Visualize side effects over time
        if len(patient_side_effects) > 1:
            st.subheader("Évolution des effets secondaires")
            
            # Ensure dates are in datetime format
            if 'report_date' in patient_side_effects.columns:
                patient_side_effects['report_date'] = pd.to_datetime(patient_side_effects['report_date'])
            
            # Standardize column names for melting
            effect_cols = {
                'eps': 'EPS',
                'akathisia': 'Akathisia',
                'weight_gain': 'Weight_Gain',
                'metabolic_changes': 'Metabolic_Changes',
                'sedation': 'Sedation',
                'sexual_dysfunction': 'Sexual_Dysfunction'
            }
            
            # Rename columns to standard names if they exist
            for old_col, new_col in effect_cols.items():
                if old_col in patient_side_effects.columns:
                    patient_side_effects[new_col] = patient_side_effects[old_col]
            
            # Determine which columns exist for melting
            value_vars = [col for col in ['EPS', 'Akathisia', 'Weight_Gain', 'Metabolic_Changes', 'Sedation', 'Sexual_Dysfunction'] 
                          if col in patient_side_effects.columns]
            
            # Prepare ID vars based on available columns
            id_vars = ['patient_id']
            date_col = 'report_date' if 'report_date' in patient_side_effects.columns else 'Date'
            id_vars.append(date_col)
            
            # Melt the data for plotting
            side_effect_long = patient_side_effects.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name='Side_Effect',
                value_name='Severity'
            )
            
            # Map variable names to French for display
            side_effect_long['Side_Effect'] = side_effect_long['Side_Effect'].map({
                'EPS': 'Sympt. extrapyramidaux',
                'Akathisia': 'Akathisie',
                'Weight_Gain': 'Prise de poids',
                'Metabolic_Changes': 'Chgt. métaboliques',
                'Sedation': 'Sédation',
                'Sexual_Dysfunction': 'Dysf. sexuelle'
            })
            
            # Create line chart
            fig = px.line(
                side_effect_long, 
                x=date_col, 
                y='Severity', 
                color='Side_Effect',
                title='Évolution des effets secondaires',
                labels={'Severity': 'Sévérité (0-10)', 'Side_Effect': 'Effet Secondaire'}
            )
            
            # Add markers to the lines
            fig.update_traces(mode='lines+markers')
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Sévérité (0-10)",
                yaxis_range=[0, 10]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a summary view
            st.subheader("Résumé des effets secondaires")
            
            # Calculate summary statistics
            summary = side_effect_long.groupby('Side_Effect')['Severity'].agg(['mean', 'max']).reset_index()
            summary.columns = ['Effet Secondaire', 'Sévérité Moyenne', 'Sévérité Maximum']
            summary['Sévérité Moyenne'] = summary['Sévérité Moyenne'].round(1)
            
            # Show summary table
            st.dataframe(summary)
            
            # Create a bar chart of max severity
            fig_max = px.bar(
                summary,
                x='Effet Secondaire',
                y='Sévérité Maximum',
                color='Effet Secondaire',
                title="Sévérité Maximum par Effet Secondaire"
            )
            st.plotly_chart(fig_max, use_container_width=True)
    else:
        st.info("Aucun effet secondaire n'a été enregistré pour ce patient.")
    
    # Form to add new side effect report
    st.subheader("Ajouter un rapport d'effets secondaires")
    with st.form("side_effect_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.now())
            eps = st.slider("Symptômes extrapyramidaux", 0, 10, 0, 
                          help="0 = Aucun, 10 = Sévères (rigidité, tremblements)")
            akathisia = st.slider("Akathisie", 0, 10, 0, 
                                help="0 = Aucune, 10 = Sévère (agitation motrice)")
            weight_gain = st.slider("Prise de poids", 0, 10, 0,
                                  help="0 = Aucune, 10 = Significative")
        
        with col2:
            metabolic_changes = st.slider("Changements métaboliques", 0, 10, 0,
                                       help="0 = Aucun, 10 = Significatifs")
            sedation = st.slider("Sédation/Somnolence", 0, 10, 0,
                              help="0 = Aucune, 10 = Sévère (affecte activités quotidiennes)")
            sexual_dysfunction = st.slider("Dysfonction sexuelle", 0, 10, 0,
                                        help="0 = Aucune, 10 = Sévère")
            
        other = st.text_input("Autres effets secondaires")
        notes = st.text_area("Notes supplémentaires")
        
        submitted = st.form_submit_button("Soumettre")
        
        # This is the second key change - using the service function to save data
        if submitted:
            # Create report data dictionary
            report_data = {
                'patient_id': st.session_state.selected_patient_id,
                'report_date': date.strftime('%Y-%m-%d'),
                'eps': eps,
                'akathisia': akathisia,
                'weight_gain': weight_gain,
                'metabolic_changes': metabolic_changes,
                'sedation': sedation,
                'sexual_dysfunction': sexual_dysfunction,
                'other_effects': other,
                'notes': notes,
                'created_by': st.session_state.get('username', 'System')
            }
            
            # Use the nurse service function to save the report
            success = save_side_effect_report(report_data)
            if success:
                st.success("Rapport d'effets secondaires enregistré avec succès.")
                st.rerun()
    
    # Add guide for recording side effects
    with st.expander("Guide pour l'évaluation des effets secondaires"):
        st.markdown("""
        ### Échelle de Sévérité des Effets Secondaires (0-10)
        
        - **0**: Aucun effet secondaire
        - **1-3**: Effets secondaires légers - N'interfèrent pas avec les activités quotidiennes
        - **4-6**: Effets secondaires modérés - Interfèrent partiellement avec les activités quotidiennes
        - **7-9**: Effets secondaires sévères - Interfèrent significativement avec les activités quotidiennes
        - **10**: Effets secondaires insupportables - Empêchent les activités quotidiennes
        
        ### Effets Secondaires Courants des Antipsychotiques
        
        - **Symptômes extrapyramidaux**: Rigidité, tremblements, mouvements anormaux
        - **Akathisie**: Agitation motrice, incapacité à rester immobile
        - **Prise de poids**: Augmentation significative du poids corporel
        - **Changements métaboliques**: Altérations de la glycémie, lipides
        - **Sédation**: Somnolence excessive, fatigue
        - **Dysfonction sexuelle**: Problèmes de libido, d'érection ou d'orgasme
        
        Si des effets secondaires sévères ou non listés surviennent, veuillez contacter immédiatement l'équipe médicale.
        """)