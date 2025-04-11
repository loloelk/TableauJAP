# components/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
import base64
import logging
import numpy as np
from services.network_analysis import generate_person_specific_network
from services.nurse_service import get_latest_nurse_inputs, get_nurse_inputs_history, get_side_effects_history

# Medication categories for First Episode Psychosis
MEDICATION_CATEGORIES = {
    'Antipsychotics_Oral': ['Olanzapine', 'Risperidone', 'Aripiprazole', 'Quetiapine', 'Clozapine'],
    'Antipsychotics_LAI': ['Aripiprazole Maintena', 'Risperdal Consta', 'Invega Sustenna'],
    'Mood_Stabilizers': ['Lithium', 'Valproate', 'Lamotrigine', 'Carbamazepine'],
    'Anxiolytics': ['Lorazepam', 'Clonazepam'],
    'Other': ['Procyclidine', 'Benztropine', 'Metformin', 'Propranolol']
}

# Helper function to get EMA data
def get_patient_ema_data(patient_id):
    """Retrieve and prepare EMA data for a specific patient"""
    if 'simulated_ema_data' not in st.session_state or st.session_state.simulated_ema_data.empty:
        logging.warning("Simulated EMA data not found in session state.")
        return pd.DataFrame()
    if 'PatientID' not in st.session_state.simulated_ema_data.columns:
         logging.error("Column 'PatientID' missing in simulated EMA data.")
         return pd.DataFrame()
    try:
        patient_ema = st.session_state.simulated_ema_data[
            st.session_state.simulated_ema_data['PatientID'] == patient_id
        ].copy()
        if 'Timestamp' in patient_ema.columns:
            patient_ema['Timestamp'] = pd.to_datetime(patient_ema['Timestamp'], errors='coerce')
            patient_ema.dropna(subset=['Timestamp'], inplace=True)
            patient_ema.sort_values(by='Timestamp', inplace=True)
        else:
            logging.warning("'Timestamp' column missing in patient EMA data.")
    except Exception as e:
         logging.error(f"Error processing EMA data for {patient_id}: {e}")
         return pd.DataFrame()
    return patient_ema

def treatment_progress(patient_ema):
    """Display treatment progress tracking based on EMA dates - adapted for psychosis timeline"""
    st.subheader("Suivi de Progression du Traitement")
    # Longer milestones for FEP treatment
    milestones = ['Éval Initiale', 'Mois 1', 'Mois 3', 'Mois 6', 'Mois 9', 'Mois 12']
    # Longer treatment duration for psychosis
    assumed_duration_days = 365  # 1 year for first episode psychosis
    
    if patient_ema.empty or 'Timestamp' not in patient_ema.columns:
        st.warning("ℹ️ Données EMA ou Timestamps manquants pour suivre la progression.")
        return
    
    try:
        first_entry = patient_ema['Timestamp'].min()
        last_entry = patient_ema['Timestamp'].max()
        days_elapsed = (last_entry - first_entry).days if pd.notna(first_entry) and pd.notna(last_entry) else 0
    except Exception as e:
         logging.error(f"Error calculating date range from EMA Timestamps: {e}")
         days_elapsed = 0
         
    progress_percentage = min((days_elapsed / assumed_duration_days) * 100, 100) if assumed_duration_days > 0 else 0
    
    # Determine milestone based on psychosis treatment timeline
    if days_elapsed <= 7: 
        current_milestone_index = 0
    elif days_elapsed <= 30: 
        current_milestone_index = 1
    elif days_elapsed <= 90: 
        current_milestone_index = 2
    elif days_elapsed <= 180: 
        current_milestone_index = 3
    elif days_elapsed <= 270: 
        current_milestone_index = 4
    else: 
        current_milestone_index = 5
        
    st.progress(progress_percentage / 100)
    st.write(f"Progression estimée: {progress_percentage:.0f}% ({days_elapsed} jours depuis la première donnée)")
    
    cols = st.columns(len(milestones))
    for i, (col, milestone) in enumerate(zip(cols, milestones)):
        with col:
            if i < current_milestone_index: 
                st.success(f"✅ {milestone}")
            elif i == current_milestone_index: 
                st.info(f"➡️ {milestone}")
            else: 
                st.markdown(f"<span style='opacity: 0.5;'>⬜ {milestone}</span>", unsafe_allow_html=True)

def patient_dashboard():
    """Main dashboard for individual patient view for psychosis clinic"""
    st.header("📊 Tableau de Bord du Patient - Clinique Premier Épisode Psychotique")
    
    if not st.session_state.get("selected_patient_id"):
        st.warning("⚠️ Aucun patient sélectionné. Veuillez en choisir un dans la barre latérale.")
        return
        
    patient_id = st.session_state.selected_patient_id
    
    if 'final_data' not in st.session_state or st.session_state.final_data.empty:
         st.error("❌ Données principales du patient non chargées.")
         return
         
    try:
         if 'ID' not in st.session_state.final_data.columns:
              st.error("Colonne 'ID' manquante dans les données patient principales.")
              return
         patient_row = st.session_state.final_data[st.session_state.final_data["ID"] == patient_id]
         if patient_row.empty:
             st.error(f"❌ Données non trouvées pour le patient {patient_id}.")
             return
         patient_data = patient_row.iloc[0]
    except Exception as e:
         st.error(f"Erreur récupération données pour {patient_id}: {e}")
         logging.exception(f"Error fetching data for patient {patient_id}")
         return

    patient_ema = get_patient_ema_data(patient_id)

    # --- Define Tabs for Psychosis Clinic ---
    tab_overview, tab_assessments, tab_hospitalizations, tab_symptoms, tab_plan, tab_side_effects, tab_notes_history = st.tabs([
        "👤 Aperçu", "📈 Évaluations", "🏥 Hospitalisations", "🧠 Symptômes", 
        "🎯 Plan de Soins", "🩺 Effets 2nd", "📝 Historique Notes"
    ])

    # --- Tab 1: Patient Overview ---
    with tab_overview:
        st.header("👤 Aperçu du Patient")
        
        # Create main layout with two columns
        left_col, right_col = st.columns([3, 2])
        
        # --- Left Column: Patient Basic Info ---
        with left_col:
            # Basic patient info
            basic_info_cols = st.columns(3)
            with basic_info_cols[0]:
                sex_numeric = patient_data.get('sexe', 'N/A')
                if str(sex_numeric) == '1': sex = "Femme"
                elif str(sex_numeric) == '2': sex = "Homme"
                else: sex = "Autre/N/A"
                st.metric(label="Sexe", value=sex)
            with basic_info_cols[1]: 
                st.metric(label="Âge", value=patient_data.get('age', 'N/A'))
            with basic_info_cols[2]: 
                # Display diagnosis rather than protocol
                st.metric(label="Diagnostic", value=patient_data.get('diagnosis', 'N/A'))
            
            # Detailed clinical data - Updated for psychosis
            with st.expander("🩺 Données Cliniques Détaillées", expanded=False):
                col1_details, col2_details = st.columns(2)
                with col1_details:
                    st.subheader("Durée de Psychose Non Traitée")
                    st.write(f"{patient_data.get('dup_months', 'N/A')} mois")
                    st.subheader("Facteurs de Risque")
                    st.write(patient_data.get('risk_factors', 'N/A'))
                with col2_details:
                    st.subheader("Consommation de Substances")
                    cannabis_use = patient_data.get('cannabis_use', 0)
                    st.write(f"Cannabis: {f'{cannabis_use} g/jour' if float(cannabis_use) > 0 else 'Non'}")
                    
                    stimulant_use = patient_data.get('stimulant_use', 0)
                    st.write(f"Stimulants: {f'{stimulant_use} pilule(s)/jour' if float(stimulant_use) > 0 else 'Non'}")
                    
                    alcohol_use = patient_data.get('alcohol_use', 0)
                    st.write(f"Alcool: {f'{alcohol_use} verre(s)/jour' if float(alcohol_use) > 0 else 'Non'}")
            
            # Medication information
            st.subheader("📋 Médications Actuelles")
            if 'medications' in patient_data.index and patient_data['medications'] != "Aucun":
                # Process medication data
                meds_list = patient_data['medications'].split('; ')
                meds_data = []
                
                for med in meds_list:
                    parts = med.split(' ')
                    if len(parts) >= 2:
                        name = ' '.join(parts[:-1])
                        dosage = parts[-1]
                        
                        # Determine medication category
                        category = "Autre"
                        for cat, meds in MEDICATION_CATEGORIES.items():
                            if name in meds:
                                category = cat
                                break
                                
                        meds_data.append({
                            "Médicament": name,
                            "Catégorie": category,
                            "Dosage": dosage
                        })
                
                if meds_data:
                    meds_df = pd.DataFrame(meds_data)
                    st.dataframe(meds_df, hide_index=True, use_container_width=True)
            else:
                st.info("Aucune médication psychiatrique n'est actuellement prescrite.")
            
            # Substance use information - Moved from hospitalizations tab
            st.markdown("---")
            st.subheader("🚬 Consommation de Substances")
            
            substance_cols = st.columns(3)
            with substance_cols[0]:
                cannabis_use = patient_data.get('cannabis_use', 0)
                st.metric("Cannabis", f"{cannabis_use} g/jour" if float(cannabis_use) > 0 else "Non")
            
            with substance_cols[1]:
                stimulant_use = patient_data.get('stimulant_use', 0)
                st.metric("Stimulants", f"{stimulant_use} pilule(s)/jour" if float(stimulant_use) > 0 else "Non")
            
            with substance_cols[2]:
                alcohol_use = patient_data.get('alcohol_use', 0)
                st.metric("Alcool", f"{alcohol_use} verre(s)/jour" if float(alcohol_use) > 0 else "Non")
            
            # Export button
            if st.button("Exporter Données Patient (CSV)"):
                try:
                    patient_main_df = patient_row.to_frame().T
                    csv = patient_main_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Télécharger (CSV)", data=csv, file_name=f"patient_{patient_id}_main_data.csv", mime='text/csv')
                except Exception as e:
                    st.error(f"Erreur export: {e}")
        
        # --- Right Column: Progress Summary for Psychosis ---
        with right_col:
            # Progress summary section - Using PANSS positive scale for psychosis
            st.subheader("📊 Résumé de Progression")
            
            # Calculate PANSS positive improvement if available
            panss_pos_bl = pd.to_numeric(patient_data.get("panss_pos_bl"), errors='coerce')
            panss_pos_fu = pd.to_numeric(patient_data.get("panss_pos_fu"), errors='coerce')
            
            if not pd.isna(panss_pos_bl) and not pd.isna(panss_pos_fu):
                delta_score = panss_pos_fu - panss_pos_bl
                
                # Set color based on improvement
                progress_color = "#4CAF50" if delta_score < 0 else "#FF5722"
                
                # Display styled progress card
                st.markdown(
                    f"""
                    <div style="
                        padding: 15px;
                        border-radius: 5px;
                        background-color: {progress_color}20;
                        border-left: 5px solid {progress_color};
                        margin-bottom: 20px;
                    ">
                        <h4 style="margin:0;">PANSS Positif: {panss_pos_bl:.0f} → {panss_pos_fu:.0f}</h4>
                        <h3 style="margin:5px 0; color: {progress_color};">
                            {delta_score:.0f} points ({((panss_pos_bl - panss_pos_fu) / panss_pos_bl * 100):.1f}% amélioration)
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display response status - Different criteria for psychosis
                if panss_pos_bl > 0:
                    pos_reduction_pct = ((panss_pos_bl - panss_pos_fu) / panss_pos_bl) * 100
                    is_responder = pos_reduction_pct >= 30  # 30% improvement in positive symptoms
                    is_remitter = panss_pos_fu <= 10  # Criteria for symptom remission
                    
                    st.markdown(f"""
                        <div style="margin-bottom: 20px;">
                            <p><strong>Statut Clinique:</strong>
                                <span style="color: {'green' if is_responder else 'red'};">
                                    {'✅ Répondeur' if is_responder else '❌ Non-répondeur'} (≥30%)
                                </span>
                                <br>
                                <span style="color: {'green' if is_remitter else 'orange'};">
                                    {'✅ Rémission Symptomatique' if is_remitter else '⚠️ Symptômes Actifs'} (≤10)
                                </span>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Données insuffisantes pour calculer la progression.")
            
            # Functional outcome - GAF for psychosis
            gaf_bl = pd.to_numeric(patient_data.get("gaf_bl"), errors='coerce')
            gaf_fu = pd.to_numeric(patient_data.get("gaf_fu"), errors='coerce')
            
            if not pd.isna(gaf_bl) and not pd.isna(gaf_fu):
                st.markdown("---")
                st.subheader("🧩 Fonctionnement Global")
                
                delta_gaf = gaf_fu - gaf_bl
                gaf_color = "#4CAF50" if delta_gaf > 0 else "#FF5722"
                
                st.markdown(
                    f"""
                    <div style="
                        padding: 15px;
                        border-radius: 5px;
                        background-color: {gaf_color}20;
                        border-left: 5px solid {gaf_color};
                        margin-bottom: 20px;
                    ">
                        <h4 style="margin:0;">GAF: {gaf_bl:.0f} → {gaf_fu:.0f}</h4>
                        <h3 style="margin:5px 0; color: {gaf_color};">
                            {delta_gaf:+.0f} points
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Interpret GAF score
                if gaf_fu >= 70:
                    st.success("✅ Bon fonctionnement social et occupationnel")
                elif gaf_fu >= 50:
                    st.info("ℹ️ Fonctionnement modéré, quelques difficultés")
                else:
                    st.warning("⚠️ Difficultés importantes dans le fonctionnement")
            
            # Hospitalization summary
            st.markdown("---")
            st.subheader("🏥 Résumé Hospitalier")
            
            hospitalizations = pd.to_numeric(patient_data.get("hospitalizations_past_year", 0), errors='coerce')
            er_visits = pd.to_numeric(patient_data.get("er_visits_past_year", 0), errors='coerce')
            
            col1_hosp, col2_hosp = st.columns(2)
            with col1_hosp:
                st.metric("Hospitalisations (12 mois)", f"{hospitalizations:.0f}")
            with col2_hosp:
                st.metric("Visites Urgence (12 mois)", f"{er_visits:.0f}")

    # --- Tab 2: Clinical Assessments for Psychosis ---
    with tab_assessments:
        st.header("📈 Évaluations Cliniques")
        subtab_panss, subtab_calgary, subtab_psyrats, subtab_cgi = st.tabs([
            "PANSS", "Calgary", "PSYRATS", "CGI"
        ])

        # PANSS subtab
        with subtab_panss:
            st.subheader("Échelle PANSS (Positive and Negative Syndrome Scale)")
            
            panss_pos_bl = pd.to_numeric(patient_data.get("panss_pos_bl"), errors='coerce')
            panss_pos_fu = pd.to_numeric(patient_data.get("panss_pos_fu"), errors='coerce')
            panss_neg_bl = pd.to_numeric(patient_data.get("panss_neg_bl"), errors='coerce')
            panss_neg_fu = pd.to_numeric(patient_data.get("panss_neg_fu"), errors='coerce')
            panss_gen_bl = pd.to_numeric(patient_data.get("panss_gen_bl"), errors='coerce')
            panss_gen_fu = pd.to_numeric(patient_data.get("panss_gen_fu"), errors='coerce')
            
            # Calculate totals
            panss_total_bl = panss_pos_bl + panss_neg_bl + panss_gen_bl if not (pd.isna(panss_pos_bl) or pd.isna(panss_neg_bl) or pd.isna(panss_gen_bl)) else np.nan
            panss_total_fu = panss_pos_fu + panss_neg_fu + panss_gen_fu if not (pd.isna(panss_pos_fu) or pd.isna(panss_neg_fu) or pd.isna(panss_gen_fu)) else np.nan
            
            # Display PANSS scores
            if pd.isna(panss_pos_bl) and pd.isna(panss_neg_bl) and pd.isna(panss_gen_bl):
                st.warning("Scores PANSS Baseline manquants.")
            else:
                col1_panss, col2_panss = st.columns(2)
                with col1_panss:
                    # Baseline scores
                    st.subheader("Scores PANSS - Baseline")
                    panss_bl_data = {
                        "Échelle": ["Positif", "Négatif", "Général", "Total"],
                        "Score": [
                            f"{panss_pos_bl:.0f}" if not pd.isna(panss_pos_bl) else "N/A",
                            f"{panss_neg_bl:.0f}" if not pd.isna(panss_neg_bl) else "N/A",
                            f"{panss_gen_bl:.0f}" if not pd.isna(panss_gen_bl) else "N/A",
                            f"{panss_total_bl:.0f}" if not pd.isna(panss_total_bl) else "N/A"
                        ]
                    }
                    panss_bl_df = pd.DataFrame(panss_bl_data)
                    st.dataframe(panss_bl_df, hide_index=True, use_container_width=True)
                    
                    # Display PANSS baseline visualization
                    if not pd.isna(panss_pos_bl) and not pd.isna(panss_neg_bl) and not pd.isna(panss_gen_bl):
                        panss_bl_chart_data = pd.DataFrame({
                            'Échelle': ['Positif', 'Négatif', 'Général'],
                            'Score': [panss_pos_bl, panss_neg_bl, panss_gen_bl]
                        })
                        fig_panss_bl = px.bar(
                            panss_bl_chart_data, 
                            x='Échelle', 
                            y='Score', 
                            title="PANSS Baseline",
                            color='Échelle',
                            color_discrete_sequence=st.session_state.PASTEL_COLORS[:3]
                        )
                        st.plotly_chart(fig_panss_bl, use_container_width=True)
                
                with col2_panss:
                    # Follow-up scores
                    st.subheader("Scores PANSS - Suivi")
                    if not pd.isna(panss_pos_fu) and not pd.isna(panss_neg_fu) and not pd.isna(panss_gen_fu):
                        panss_fu_data = {
                            "Échelle": ["Positif", "Négatif", "Général", "Total"],
                            "Score": [
                                f"{panss_pos_fu:.0f}",
                                f"{panss_neg_fu:.0f}",
                                f"{panss_gen_fu:.0f}",
                                f"{panss_total_fu:.0f}" if not pd.isna(panss_total_fu) else "N/A"
                            ],
                            "Δ": [
                                f"{panss_pos_fu - panss_pos_bl:+.0f}" if not pd.isna(panss_pos_bl) and not pd.isna(panss_pos_fu) else "N/A",
                                f"{panss_neg_fu - panss_neg_bl:+.0f}" if not pd.isna(panss_neg_bl) and not pd.isna(panss_neg_fu) else "N/A",
                                f"{panss_gen_fu - panss_gen_bl:+.0f}" if not pd.isna(panss_gen_bl) and not pd.isna(panss_gen_fu) else "N/A",
                                f"{panss_total_fu - panss_total_bl:+.0f}" if not pd.isna(panss_total_bl) and not pd.isna(panss_total_fu) else "N/A"
                            ]
                        }
                        panss_fu_df = pd.DataFrame(panss_fu_data)
                        st.dataframe(panss_fu_df, hide_index=True, use_container_width=True)
                        
                        # Display PANSS comparison visualization
                        panss_comp_data = pd.DataFrame({
                            'Échelle': ['Positif', 'Négatif', 'Général', 'Total'] * 2,
                            'Temps': ['Baseline'] * 4 + ['Suivi'] * 4,
                            'Score': [
                                panss_pos_bl, panss_neg_bl, panss_gen_bl, panss_total_bl,
                                panss_pos_fu, panss_neg_fu, panss_gen_fu, panss_total_fu
                            ]
                        })
                        fig_panss_comp = px.bar(
                            panss_comp_data,
                            x='Échelle',
                            y='Score',
                            color='Temps',
                            barmode='group',
                            title="PANSS: Baseline vs Suivi",
                            color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                        )
                        st.plotly_chart(fig_panss_comp, use_container_width=True)
                    else:
                        st.info("Données de suivi PANSS non disponibles.")
                
                # Calculate and display percent changes
                if not pd.isna(panss_pos_bl) and not pd.isna(panss_pos_fu) and panss_pos_bl > 0:
                    st.markdown("---")
                    st.subheader("Pourcentage d'Amélioration")
                    
                    pos_change_pct = ((panss_pos_bl - panss_pos_fu) / panss_pos_bl) * 100
                    neg_change_pct = ((panss_neg_bl - panss_neg_fu) / panss_neg_bl) * 100 if panss_neg_bl > 0 else 0
                    gen_change_pct = ((panss_gen_bl - panss_gen_fu) / panss_gen_bl) * 100 if panss_gen_bl > 0 else 0
                    total_change_pct = ((panss_total_bl - panss_total_fu) / panss_total_bl) * 100 if panss_total_bl > 0 else 0
                    
                    # Create metrics columns
                    change_cols = st.columns(4)
                    with change_cols[0]:
                        st.metric("Positif", f"{pos_change_pct:.1f}%", 
                                delta="Répondeur" if pos_change_pct >= 30 else "Non-répondeur")
                    with change_cols[1]:
                        st.metric("Négatif", f"{neg_change_pct:.1f}%")
                    with change_cols[2]:
                        st.metric("Général", f"{gen_change_pct:.1f}%")
                    with change_cols[3]:
                        st.metric("Total", f"{total_change_pct:.1f}%")
                    
                    # Add interpretation
                    st.info("ℹ️ Une amélioration de ≥30% des symptômes positifs est considérée comme une réponse cliniquement significative.")

        # Calgary Depression Scale
        with subtab_calgary:
            st.subheader("Échelle de Dépression de Calgary")
            
            calgary_bl = pd.to_numeric(patient_data.get("calgary_bl"), errors='coerce')
            calgary_fu = pd.to_numeric(patient_data.get("calgary_fu"), errors='coerce')
            
            if pd.isna(calgary_bl):
                st.warning("Score Calgary Baseline manquant.")
            else:
                cols_calgary = st.columns(2)
                with cols_calgary[0]:
                    st.metric("Calgary Baseline", f"{calgary_bl:.0f}")
                    
                    # Interpret Calgary score
                    if calgary_bl <= 5:
                        severity = "Légère/Absente"
                    elif calgary_bl <= 10:
                        severity = "Modérée"
                    else:
                        severity = "Sévère"
                    st.write(f"**Sévérité Dépressive:** {severity}")
                    
                    if not pd.isna(calgary_fu):
                        delta_calgary = calgary_fu - calgary_bl
                        st.metric("Calgary Suivi", f"{calgary_fu:.0f}", delta=f"{delta_calgary:+.0f} points")
                    else:
                        st.metric("Calgary Suivi", "N/A")
                
                with cols_calgary[1]:
                    # Create visualization for Calgary scores
                    if not pd.isna(calgary_fu):
                        calgary_data = pd.DataFrame({
                            'Évaluation': ['Baseline', 'Suivi'],
                            'Score': [calgary_bl, calgary_fu]
                        })
                        fig_calgary = px.bar(
                            calgary_data,
                            x='Évaluation',
                            y='Score',
                            title="Évolution Score Calgary",
                            color='Évaluation',
                            color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                        )
                        # Add threshold line for clinical significance
                        fig_calgary.add_hline(y=5, line_dash="dash", line_color="red", 
                                           annotation_text="Seuil clinique", annotation_position="bottom right")
                        st.plotly_chart(fig_calgary, use_container_width=True)
                    
                # Add item-level analysis if available
                st.markdown("---")
                st.subheader("Analyse par Item")
                
                # Check if item-level data exists
                items_available = True
                for i in range(1, 10):  # Assuming Calgary has 9 items
                    if f'calgary_item{i}_bl' not in patient_data.index:
                        items_available = False
                        break
                
                if items_available:
                    # Display item-level analysis
                    calgary_items_data = []
                    for i in range(1, 10):
                        bl_col = f'calgary_item{i}_bl'
                        fu_col = f'calgary_item{i}_fu'
                        
                        # Use Calgary items mapping from session state if available
                        item_label = st.session_state.get('CALGARY_ITEMS_MAPPING', {}).get(str(i), f"Item {i}")
                        
                        bl_val = pd.to_numeric(patient_data.get(bl_col), errors='coerce')
                        fu_val = pd.to_numeric(patient_data.get(fu_col), errors='coerce')
                        
                        calgary_items_data.append({
                            'Item': item_label,
                            'Baseline': bl_val,
                            'Suivi': fu_val
                        })
                    
                    # Create item-level dataframe and visualization
                    calgary_items_df = pd.DataFrame(calgary_items_data)
                    calgary_items_long = calgary_items_df.melt(
                        id_vars='Item', 
                        var_name='Temps', 
                        value_name='Score'
                    ).dropna(subset=['Score'])
                    
                    fig_calgary_items = px.bar(
                        calgary_items_long,
                        x='Item',
                        y='Score',
                        color='Temps',
                        barmode='group',
                        title="Scores par Item - Calgary",
                        color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                    )
                    fig_calgary_items.update_xaxes(tickangle=-45)
                    fig_calgary_items.update_yaxes(range=[0, 4])  # Calgary items are 0-3
                    
                    st.plotly_chart(fig_calgary_items, use_container_width=True)
                else:
                    st.info("Données détaillées par item non disponibles pour l'échelle Calgary.")

        # PSYRATS subtab
        with subtab_psyrats:
            st.subheader("PSYRATS (Psychotic Symptom Rating Scales)")
            
            # Create subtabs for hallucinations and delusions
            psyrats_ah_tab, psyrats_del_tab = st.tabs(["Hallucinations Auditives", "Délusions"])
            
            # Hallucinations tab
            with psyrats_ah_tab:
                psyrats_ah_bl = pd.to_numeric(patient_data.get("psyrats_ah_bl"), errors='coerce')
                psyrats_ah_fu = pd.to_numeric(patient_data.get("psyrats_ah_fu"), errors='coerce')
                
                if pd.isna(psyrats_ah_bl):
                    st.info("Patient sans hallucinations auditives rapportées à l'évaluation initiale.")
                else:
                    st.subheader("Hallucinations Auditives")
                    
                    cols_ah = st.columns(2)
                    with cols_ah[0]:
                        st.metric("PSYRATS-AH Baseline", f"{psyrats_ah_bl:.0f}")
                        if not pd.isna(psyrats_ah_fu):
                            delta_ah = psyrats_ah_fu - psyrats_ah_bl
                            improvement_pct = ((psyrats_ah_bl - psyrats_ah_fu) / psyrats_ah_bl * 100) if psyrats_ah_bl > 0 else 0
                            st.metric("PSYRATS-AH Suivi", f"{psyrats_ah_fu:.0f}", delta=f"{delta_ah:+.0f} points")
                            st.metric("Amélioration", f"{improvement_pct:.1f}%")
                    
                    with cols_ah[1]:
                        # Visualization for hallucinations
                        if not pd.isna(psyrats_ah_fu):
                            ah_data = pd.DataFrame({
                                'Évaluation': ['Baseline', 'Suivi'],
                                'Score': [psyrats_ah_bl, psyrats_ah_fu]
                            })
                            fig_ah = px.bar(
                                ah_data,
                                x='Évaluation',
                                y='Score',
                                title="Évolution Hallucinations Auditives",
                                color='Évaluation',
                                color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                            )
                            st.plotly_chart(fig_ah, use_container_width=True)
                    
                    # Item-level analysis if available
                    ah_items_available = True
                    for i in range(1, 12):  # PSYRATS-AH has 11 items
                        if f'psyrats_ah_item{i}_bl' not in patient_data.index:
                            ah_items_available = False
                            break
                    
                    if ah_items_available:
                        st.markdown("---")
                        st.subheader("Caractéristiques des Hallucinations")
                        
                        ah_items_data = []
                        for i in range(1, 12):
                            bl_col = f'psyrats_ah_item{i}_bl'
                            fu_col = f'psyrats_ah_item{i}_fu'
                            
                            # Use PSYRATS items mapping if available
                            item_label = st.session_state.get('PSYRATS_AH_ITEMS_MAPPING', {}).get(str(i), f"Item {i}")
                            
                            bl_val = pd.to_numeric(patient_data.get(bl_col), errors='coerce')
                            fu_val = pd.to_numeric(patient_data.get(fu_col), errors='coerce')
                            
                            ah_items_data.append({
                                'Item': item_label,
                                'Baseline': bl_val,
                                'Suivi': fu_val
                            })
                        
                        ah_items_df = pd.DataFrame(ah_items_data)
                        ah_items_long = ah_items_df.melt(
                            id_vars='Item', 
                            var_name='Temps', 
                            value_name='Score'
                        ).dropna(subset=['Score'])
                        
                        fig_ah_items = px.bar(
                            ah_items_long,
                            x='Item',
                            y='Score',
                            color='Temps',
                            barmode='group',
                            title="Composantes des Hallucinations",
                            color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                        )
                        fig_ah_items.update_xaxes(tickangle=-45)
                        fig_ah_items.update_yaxes(range=[0, 4])  # PSYRATS items are 0-4
                        
                        st.plotly_chart(fig_ah_items, use_container_width=True)
                    else:
                        st.info("Données détaillées par item non disponibles pour les hallucinations.")
            
            # Delusions tab
            with psyrats_del_tab:
                psyrats_del_bl = pd.to_numeric(patient_data.get("psyrats_del_bl"), errors='coerce')
                psyrats_del_fu = pd.to_numeric(patient_data.get("psyrats_del_fu"), errors='coerce')
                
                if pd.isna(psyrats_del_bl):
                    st.info("Patient sans délusions rapportées à l'évaluation initiale.")
                else:
                    st.subheader("Délusions")
                    
                    cols_del = st.columns(2)
                    with cols_del[0]:
                        st.metric("PSYRATS-DEL Baseline", f"{psyrats_del_bl:.0f}")
                        if not pd.isna(psyrats_del_fu):
                            delta_del = psyrats_del_fu - psyrats_del_bl
                            improvement_pct = ((psyrats_del_bl - psyrats_del_fu) / psyrats_del_bl * 100) if psyrats_del_bl > 0 else 0
                            st.metric("PSYRATS-DEL Suivi", f"{psyrats_del_fu:.0f}", delta=f"{delta_del:+.0f} points")
                            st.metric("Amélioration", f"{improvement_pct:.1f}%")
                    
                    with cols_del[1]:
                        # Visualization for delusions
                        if not pd.isna(psyrats_del_fu):
                            del_data = pd.DataFrame({
                                'Évaluation': ['Baseline', 'Suivi'],
                                'Score': [psyrats_del_bl, psyrats_del_fu]
                            })
                            fig_del = px.bar(
                                del_data,
                                x='Évaluation',
                                y='Score',
                                title="Évolution des Délusions",
                                color='Évaluation',
                                color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                            )
                            st.plotly_chart(fig_del, use_container_width=True)
                    
                    # Item-level analysis if available
                    del_items_available = True
                    for i in range(1, 7):  # PSYRATS-DEL has 6 items
                        if f'psyrats_del_item{i}_bl' not in patient_data.index:
                            del_items_available = False
                            break
                    
                    if del_items_available:
                        st.markdown("---")
                        st.subheader("Caractéristiques des Délusions")
                        
                        del_items_data = []
                        for i in range(1, 7):
                            bl_col = f'psyrats_del_item{i}_bl'
                            fu_col = f'psyrats_del_item{i}_fu'
                            
                            # Use PSYRATS items mapping if available
                            item_label = st.session_state.get('PSYRATS_DEL_ITEMS_MAPPING', {}).get(str(i), f"Item {i}")
                            
                            bl_val = pd.to_numeric(patient_data.get(bl_col), errors='coerce')
                            fu_val = pd.to_numeric(patient_data.get(fu_col), errors='coerce')
                            
                            del_items_data.append({
                                'Item': item_label,
                                'Baseline': bl_val,
                                'Suivi': fu_val
                            })
                        
                        del_items_df = pd.DataFrame(del_items_data)
                        del_items_long = del_items_df.melt(
                            id_vars='Item', 
                            var_name='Temps', 
                            value_name='Score'
                        ).dropna(subset=['Score'])
                        
                        fig_del_items = px.bar(
                            del_items_long,
                            x='Item',
                            y='Score',
                            color='Temps',
                            barmode='group',
                            title="Composantes des Délusions",
                            color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                        )
                        fig_del_items.update_xaxes(tickangle=-45)
                        fig_del_items.update_yaxes(range=[0, 4])  # PSYRATS items are 0-4
                        
                        st.plotly_chart(fig_del_items, use_container_width=True)
                    else:
                        st.info("Données détaillées par item non disponibles pour les délusions.")
                        
        # CGI subtab
        with subtab_cgi:
            st.subheader("Impression Clinique Globale (CGI)")
            
            cgi_s_bl = pd.to_numeric(patient_data.get("cgi_s_bl"), errors='coerce')
            cgi_s_fu = pd.to_numeric(patient_data.get("cgi_s_fu"), errors='coerce')
            cgi_i = pd.to_numeric(patient_data.get("cgi_i"), errors='coerce')
            
            if pd.isna(cgi_s_bl):
                st.warning("Score CGI-S Baseline manquant.")
            else:
                # Define CGI interpretation
                cgi_severity = {
                    1: "Normal, pas du tout malade",
                    2: "À la limite",
                    3: "Légèrement malade",
                    4: "Modérément malade",
                    5: "Manifestement malade",
                    6: "Gravement malade",
                    7: "Parmi les patients les plus malades"
                }
                
                cgi_improvement = {
                    1: "Très fortement amélioré",
                    2: "Fortement amélioré",
                    3: "Légèrement amélioré",
                    4: "Pas de changement",
                    5: "Légèrement aggravé",
                    6: "Fortement aggravé",
                    7: "Très fortement aggravé"
                }
                
                cols_cgi = st.columns(3)
                
                with cols_cgi[0]:
                    st.metric("CGI-S Baseline", f"{cgi_s_bl:.0f}")
                    st.markdown(f"**Interprétation:** {cgi_severity.get(int(cgi_s_bl), 'N/A')}")
                
                with cols_cgi[1]:
                    if not pd.isna(cgi_s_fu):
                        st.metric("CGI-S Suivi", f"{cgi_s_fu:.0f}")
                        st.markdown(f"**Interprétation:** {cgi_severity.get(int(cgi_s_fu), 'N/A')}")
                    else:
                        st.metric("CGI-S Suivi", "N/A")
                
                with cols_cgi[2]:
                    if not pd.isna(cgi_i):
                        st.metric("CGI-I (Amélioration)", f"{cgi_i:.0f}")
                        st.markdown(f"**Interprétation:** {cgi_improvement.get(int(cgi_i), 'N/A')}")
                    else:
                        st.metric("CGI-I", "N/A")
                
                # CGI visualization
                st.markdown("---")
                
                if not pd.isna(cgi_s_fu):
                    # Create CGI-S comparison chart
                    cgi_s_data = pd.DataFrame({
                        'Évaluation': ['Baseline', 'Suivi'],
                        'Score': [cgi_s_bl, cgi_s_fu]
                    })
                    
                    fig_cgi_s = px.bar(
                        cgi_s_data,
                        x='Évaluation',
                        y='Score',
                        title="Évolution CGI-Sévérité",
                        color='Évaluation',
                        color_discrete_sequence=st.session_state.PASTEL_COLORS[:2]
                    )
                    fig_cgi_s.update_yaxes(range=[1, 7], dtick=1)  # CGI is 1-7
                    
                    st.plotly_chart(fig_cgi_s, use_container_width=True)
                    
                    # Add CGI interpretation table
                    st.subheader("Guide d'Interprétation CGI")
                    
                    cgi_guide = pd.DataFrame({
                        'Score': list(range(1, 8)),
                        'CGI-Sévérité': [cgi_severity[i] for i in range(1, 8)],
                        'CGI-Amélioration': [cgi_improvement[i] for i in range(1, 8)]
                    })
                    
                    st.dataframe(cgi_guide, hide_index=True, use_container_width=True)
                    
                else:
                    st.info("Données CGI de suivi non disponibles.")

    # --- Tab 3: Hospitalizations and Healthcare Utilization ---
    with tab_hospitalizations:
        st.header("🏥 Hospitalisations et Utilisation des Services")
        
        # Display metrics
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            hospitalizations = pd.to_numeric(patient_data.get("hospitalizations_past_year", 0), errors='coerce')
            st.metric("Hospitalisations (12 mois)", f"{hospitalizations:.0f}")
        
        with metric_cols[1]:
            er_visits = pd.to_numeric(patient_data.get("er_visits_past_year", 0), errors='coerce')
            st.metric("Visites Urgence (12 mois)", f"{er_visits:.0f}")
        
        with metric_cols[2]:
            hosp_days = pd.to_numeric(patient_data.get("total_hospital_days", 0), errors='coerce')
            st.metric("Jours d'hospitalisation (Total)", f"{hosp_days:.0f}")
        
        # Timeline visualization
        st.markdown("---")
        st.subheader("Chronologie des Événements Cliniques")
        
        # Create a timeline from available data
        # This is a simplified example - in a real implementation, you would have a proper events table
        timeline_events = []
        
        # Add baseline assessment
        if 'baseline_date' in patient_data:
            timeline_events.append({
                'Date': pd.to_datetime(patient_data['baseline_date']),
                'Événement': 'Évaluation Initiale',
                'Type': 'Assessment'
            })
        else:
            # Use a default date if baseline_date is not available
            baseline_date = pd.to_datetime('today') - pd.Timedelta(days=180)
            timeline_events.append({
                'Date': baseline_date,
                'Événement': 'Évaluation Initiale (estimée)',
                'Type': 'Assessment'
            })
        
        # Add hospitalizations (simplified example)
        # In a real implementation, you would have actual hospitalization records
        if hospitalizations > 0:
            # Create synthetic hospitalization events spread over the past year
            for i in range(int(hospitalizations)):
                days_ago = 30 + (330 // hospitalizations) * i
                hosp_date = pd.to_datetime('today') - pd.Timedelta(days=days_ago)
                timeline_events.append({
                    'Date': hosp_date,
                    'Événement': f'Hospitalisation #{i+1}',
                    'Type': 'Hospitalization'
                })
        
        # Add follow-up assessment
        if 'followup_date' in patient_data:
            timeline_events.append({
                'Date': pd.to_datetime(patient_data['followup_date']),
                'Événement': 'Évaluation de Suivi',
                'Type': 'Assessment'
            })
        
        # Create a dataframe and sort events
        if timeline_events:
            timeline_df = pd.DataFrame(timeline_events)
            timeline_df = timeline_df.sort_values('Date')
            
            # Set colors by event type
            color_map = {
                'Assessment': '#4CAF50',
                'Hospitalization': '#F44336'
            }
            
            # Create timeline visualization
            fig_timeline = px.scatter(
                timeline_df, 
                x='Date', 
                y=[1] * len(timeline_df),
                color='Type',
                text='Événement',
                color_discrete_map=color_map,
                title="Chronologie des Événements",
                height=300
            )
            
            # Customize the timeline
            fig_timeline.update_traces(marker=dict(size=12), mode='markers+text', textposition='top center')
            fig_timeline.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            fig_timeline.update_layout(showlegend=True)
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Données insuffisantes pour générer une chronologie.")
        
        # Display note about data simulation
        st.markdown("""
        > **Note:** Cette visualisation est basée sur des données simulées. Dans une implémentation réelle,
        > la chronologie contiendrait des données précises sur les hospitalisations, les visites aux urgences,
        > les consultations et les changements de médication.
        """)

    # --- Tab 4: Symptoms Monitoring (EMA data) ---
    with tab_symptoms:
        st.header("🧠 Suivi des Symptômes Psychotiques")
        
        if patient_ema.empty:
            st.info("ℹ️ Aucune donnée EMA disponible pour ce patient.")
        else:
            try:
                # Treatment progress display
                treatment_progress(patient_ema)
                st.markdown("---")
                
                if 'Day' not in patient_ema.columns:
                    st.warning("Colonne 'Day' manquante dans les données EMA.")
                else:
                    patient_ema['Day'] = pd.to_numeric(patient_ema['Day'], errors='coerce').dropna().astype(int)
                    st.subheader("📉 Évolution des Symptômes Quotidiens")
                    
                    # Check for symptom columns
                    if 'SYMPTOMS' not in st.session_state:
                        st.error("Erreur: Liste des symptômes EMA non définie.")
                        daily_symptoms = pd.DataFrame()
                        available_categories = {}
                    else:
                        symptoms_present = [s for s in st.session_state.SYMPTOMS if s in patient_ema.columns]
                        
                        if not symptoms_present:
                            st.warning("Aucune colonne de symptôme EMA trouvée.")
                            daily_symptoms = pd.DataFrame()
                            available_categories = {}
                        else:
                            # Filter for numeric columns
                            numeric_cols = patient_ema[symptoms_present].select_dtypes(include=np.number).columns.tolist()
                            
                            if not numeric_cols:
                                st.warning("Aucune colonne de symptôme EMA numérique trouvée.")
                                daily_symptoms = pd.DataFrame()
                                available_categories = {}
                            else:
                                # Calculate daily averages
                                try:
                                    daily_symptoms = patient_ema.groupby('Day')[numeric_cols].mean().reset_index()
                                    
                                    # Define symptom categories for psychosis
                                    symptom_categories = {
                                        "Symptômes Positifs": [s for s in st.session_state.get('POSITIVE_SYMPTOMS', []) if s in numeric_cols],
                                        "Symptômes Négatifs": [s for s in st.session_state.get('NEGATIVE_SYMPTOMS', []) if s in numeric_cols],
                                        "Symptômes Affectifs": [s for s in st.session_state.get('AFFECTIVE_SYMPTOMS', []) if s in numeric_cols],
                                        "Autres": [s for s in numeric_cols if s not in 
                                                  st.session_state.get('POSITIVE_SYMPTOMS', []) + 
                                                  st.session_state.get('NEGATIVE_SYMPTOMS', []) + 
                                                  st.session_state.get('AFFECTIVE_SYMPTOMS', [])]
                                    }
                                    
                                    available_categories = {k: v for k, v in symptom_categories.items() if v}
                                except Exception as e:
                                    st.error(f"Erreur calcul moyennes: {e}")
                                    logging.exception(f"Error daily means {patient_id}")
                                    daily_symptoms = pd.DataFrame()
                                    available_categories = {}
                    
                    # Display trend visualization if data available
                    if not daily_symptoms.empty and available_categories:
                        selected_category = st.selectbox(
                            "Afficher tendance:", 
                            list(available_categories.keys()), 
                            key="ema_cat_symptoms"
                        )
                        selected_symptoms = available_categories[selected_category]
                        
                        fig_symptoms = px.line(
                            daily_symptoms, 
                            x="Day", 
                            y=selected_symptoms, 
                            markers=True, 
                            title=f"Tendance: {selected_category}", 
                            template="plotly_white", 
                            labels={"value": "Sévérité (0-10)", "variable": "Symptôme"}
                        )
                        st.plotly_chart(fig_symptoms, use_container_width=True)
                        
                        # Add variability analysis
                        st.markdown("---")
                        st.subheader("📈 Variabilité des Symptômes")
                        st.info("La variabilité des symptômes peut indiquer des fluctuations dans l'état clinique ou des facteurs de stress.")
                        
                        rolling_window = st.slider("Fenêtre d'analyse (jours)", 3, 14, 7, key="symptom_var_window")
                        
                        if len(daily_symptoms) < rolling_window:
                            st.warning(f"Pas assez de jours ({len(daily_symptoms)}) pour cette fenêtre d'analyse.")
                        else:
                            try:
                                variability_df = daily_symptoms[['Day'] + selected_symptoms].copy()
                                
                                # Calculate rolling standard deviation
                                for symptom in selected_symptoms:
                                    variability_df[symptom] = variability_df[symptom].rolling(
                                        window=rolling_window, 
                                        min_periods=max(2, rolling_window // 2)
                                    ).std()
                                
                                variability_df.dropna(inplace=True)
                                
                                if not variability_df.empty:
                                    fig_variability = px.line(
                                        variability_df, 
                                        x='Day', 
                                        y=selected_symptoms, 
                                        title=f"Variabilité ({rolling_window}j): {selected_category}", 
                                        template="plotly_white", 
                                        labels={"value": f"Écart-Type ({rolling_window}j)", "variable": "Symptôme"}
                                    )
                                    fig_variability.update_layout(yaxis_range=[0, None])
                                    st.plotly_chart(fig_variability, use_container_width=True)
                                else:
                                    st.info(f"Pas assez de données pour calculer la variabilité avec cette fenêtre.")
                            except Exception as e:
                                st.error(f"Erreur calcul variabilité: {e}")
                                logging.exception(f"Error variability {patient_id}")
                        
                        # Optional: Correlation heatmap
                        st.markdown("---")
                        if st.checkbox("Afficher matrice de corrélation", key="show_symptom_corr"):
                            st.subheader("↔️ Corrélations Entre Symptômes")
                            
                            numeric_ema_cols = patient_ema[selected_symptoms].select_dtypes(include=np.number).columns.tolist()
                            
                            if len(numeric_ema_cols) < 2:
                                st.warning("Au moins deux symptômes sont nécessaires pour calculer des corrélations.")
                            else:
                                try:
                                    corr_matrix = patient_ema[numeric_ema_cols].corr()
                                    
                                    fig_heatmap = px.imshow(
                                        corr_matrix, 
                                        text_auto=".2f", 
                                        aspect="auto", 
                                        color_continuous_scale="RdBu_r",
                                        title=f"Corrélations: {selected_category}"
                                    )
                                    st.plotly_chart(fig_heatmap, use_container_width=True)
                                    
                                    with st.expander("💡 Interprétation des corrélations"):
                                        st.markdown("""
                                        - **Corrélation positive (bleu)** : Les symptômes augmentent ou diminuent ensemble
                                        - **Corrélation négative (rouge)** : Quand un symptôme augmente, l'autre diminue
                                        - **Valeurs proches de 0** : Peu ou pas de relation entre les symptômes
                                        
                                        Une forte corrélation entre certains symptômes peut suggérer des mécanismes sous-jacents communs
                                        ou des facteurs déclenchants similaires.
                                        """)
                                except Exception as e:
                                    st.error(f"Erreur génération heatmap: {e}")
                                    logging.exception(f"Error heatmap {patient_id}")
                    else:
                        st.info("Données insuffisantes pour visualiser les tendances des symptômes.")
                
                # Display network analysis option if enabled
                st.markdown("---")
                st.subheader("🕸️ Analyse Réseau de Symptômes")
                
                if patient_ema.empty:
                    st.warning("⚠️ Données EMA non disponibles pour l'analyse réseau.")
                elif len(patient_ema) < 10:
                    st.warning(f"⚠️ Pas assez de données EMA ({len(patient_ema)}) pour une analyse réseau fiable.")
                else:
                    st.info("Cette analyse montre les interactions potentielles entre les symptômes psychotiques au fil du temps.")
                    
                    threshold = st.slider("Seuil de connexion", 0.05, 0.5, 0.15, 0.05, key="network_thresh")
                    
                    if st.button("🔄 Générer/Actualiser Réseau"):
                        try:
                            if 'SYMPTOMS' not in st.session_state:
                                st.error("Erreur: Liste des symptômes EMA non définie.")
                            else:
                                symptoms_available = [s for s in st.session_state.SYMPTOMS if s in patient_ema.columns]
                                
                                if not symptoms_available:
                                    st.error("❌ Aucun symptôme valide trouvé dans les données.")
                                else:
                                    fig_network = generate_person_specific_network(
                                        patient_ema, 
                                        patient_id, 
                                        symptoms_available, 
                                        threshold=threshold
                                    )
                                    st.plotly_chart(fig_network, use_container_width=True)
                                    
                                    with st.expander("💡 Interprétation du réseau"):
                                        st.markdown("""
                                        Ce réseau montre comment les symptômes s'influencent mutuellement au fil du temps:
                                        
                                        - Chaque **nœud** représente un symptôme
                                        - Les **connexions** (arêtes) indiquent une influence temporelle entre symptômes
                                        - La **couleur** des connexions indique la direction: rouge = positive, bleue = négative
                                        - L'**épaisseur** des connexions reflète la force de l'influence
                                        
                                        Ces informations peuvent aider à identifier les symptômes "centraux" qui pourraient être des cibles 
                                        d'intervention prioritaires.
                                        """)
                        except Exception as e:
                            st.error(f"❌ Erreur génération réseau: {e}")
                            logging.exception(f"Network gen failed {patient_id}")
                    else:
                        st.info("Cliquez sur le bouton pour générer l'analyse réseau.")
            except Exception as e:
                st.error(f"Erreur analyse symptômes: {e}")
                logging.exception(f"Error symptom analysis {patient_id}")

    # --- Tab 5: Treatment Plan (no major changes needed) ---
    with tab_plan:
        st.header("🎯 Plan de Soins Actuel")
        st.info("Affiche la **dernière** entrée. Pour ajouter/modifier, allez à 'Plan de Soins et Entrées Infirmières'.")
        try:
             latest_plan = get_latest_nurse_inputs(patient_id)
             if latest_plan and latest_plan.get('timestamp'):
                 plan_date = pd.to_datetime(latest_plan.get('timestamp')).strftime('%Y-%m-%d %H:%M'); created_by = latest_plan.get('created_by', 'N/A')
                 st.subheader(f"Dernière MàJ: {plan_date} (par {created_by})")
                 col_stat, col_symp, col_int = st.columns([1,2,2])
                 with col_stat: st.metric("Statut Objectif", latest_plan.get('goal_status', 'N/A'))
                 with col_symp: st.markdown(f"**Sympt. Cibles:**\n> {latest_plan.get('target_symptoms', 'N/A')}")
                 with col_int: st.markdown(f"**Interv. Planifiées:**\n> {latest_plan.get('planned_interventions', 'N/A')}")
                 st.markdown("---"); st.markdown(f"**Objectifs:**\n_{latest_plan.get('objectives', 'N/A')}_"); st.markdown(f"**Tâches:**\n_{latest_plan.get('tasks', 'N/A')}_"); st.markdown(f"**Commentaires:**\n_{latest_plan.get('comments', 'N/A')}_")
             elif latest_plan: st.warning("Dernier plan trouvé mais date inconnue.")
             else: st.warning(f"ℹ️ Aucun plan trouvé pour {patient_id}.")
        except Exception as e: st.error(f"Erreur chargement plan: {e}"); logging.exception(f"Error loading care plan {patient_id}")

    # --- Tab 6: Side Effects (updated for antipsychotics) ---
    with tab_side_effects:
        st.header("🩺 Suivi Effets Secondaires (Résumé)")
        st.info("💡 Résumé des effets secondaires des antipsychotiques. Pour détails/ajout, voir page dédiée.")
        try:
            # Get side effects data
            side_effects_history = get_side_effects_history(patient_id)
            
            if not side_effects_history.empty:
                st.subheader("Effets Secondaires Signalés")
                
                # Update side effect columns for antipsychotics
                severity_cols = []
                for col in ['eps', 'akathisia', 'weight_gain', 'metabolic_changes', 'sedation', 'sexual_dysfunction']:
                    if col in side_effects_history.columns:
                        severity_cols.append(col)
                
                if not severity_cols:
                    st.warning("Aucun effet secondaire trouvé dans les données.")
                else:
                    summary_list = []
                    for col in severity_cols:
                        # Convert column to numeric
                        side_effects_history[col] = pd.to_numeric(side_effects_history[col], errors='coerce').fillna(0)
                        # Count non-zero values
                        count = (side_effects_history[col] > 0).sum()
                        if count > 0:
                            max_sev = side_effects_history[col].max()
                            summary_list.append(f"{col.replace('_', ' ').capitalize()}: {count}x (max {max_sev:.0f}/10)")
                    
                    # Show the summary list
                    if summary_list:
                        st.markdown("- " + "\n- ".join(summary_list))
                    else:
                        st.info("Aucun effet secondaire (> 0) signalé.")
                    
                    # Add visualization of side effects over time
                    if len(side_effects_history) > 1:
                        st.subheader("Évolution des Effets Secondaires")
                        
                        # Ensure dates are properly formatted
                        date_col = 'report_date'
                        if date_col in side_effects_history.columns:
                            side_effects_history[date_col] = pd.to_datetime(side_effects_history[date_col])
                            
                            # Prepare data for plotting
                            plot_data = side_effects_history.melt(
                                id_vars=[date_col],
                                value_vars=severity_cols,
                                var_name='Side_Effect',
                                value_name='Severity'
                            )
                            
                            # Map names to French
                            name_map = {
                                'eps': 'Symp. extrapyramidaux',
                                'akathisia': 'Akathisie',
                                'weight_gain': 'Prise de poids',
                                'metabolic_changes': 'Chgt. métaboliques',
                                'sedation': 'Sédation',
                                'sexual_dysfunction': 'Dysf. sexuelle'
                            }
                            plot_data['Side_Effect'] = plot_data['Side_Effect'].map(lambda x: name_map.get(x, x.capitalize()))
                            
                            # Create line chart
                            fig = px.line(
                                plot_data, 
                                x=date_col, 
                                y='Severity', 
                                color='Side_Effect',
                                title='Évolution des Effets Secondaires',
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
                            
                            # Create a summary statistics visualization
                            st.subheader("Résumé par Effet Secondaire")
                            
                            # Calculate summary statistics
                            summary = plot_data.groupby('Side_Effect')['Severity'].agg(['mean', 'max']).reset_index()
                            summary.columns = ['Effet Secondaire', 'Moyenne', 'Maximum']
                            summary['Moyenne'] = summary['Moyenne'].round(1)
                            
                            # Create a bar chart of max severity
                            fig_max = px.bar(
                                summary,
                                x='Effet Secondaire',
                                y='Maximum',
                                color='Effet Secondaire',
                                title="Sévérité Maximum par Effet Secondaire"
                            )
                            fig_max.update_xaxes(tickangle=-45)
                            st.plotly_chart(fig_max, use_container_width=True)

                    # Show the latest report details
                    if len(side_effects_history) > 0:
                        latest_report = side_effects_history.iloc[0]
                        
                        # Get notes and other effects
                        latest_note = latest_report.get('notes', '') if 'notes' in latest_report else ''
                        latest_other = latest_report.get('other_effects', '') if 'other_effects' in latest_report else ''
                        
                        # Get report date with error handling
                        report_date = "Inconnue"
                        if 'report_date' in latest_report and pd.notna(latest_report['report_date']):
                            try:
                                report_date = pd.to_datetime(latest_report['report_date']).strftime('%Y-%m-%d')
                            except:
                                pass
                        
                        # Show details if available
                        if latest_note or latest_other:
                            with st.expander(f"Détails Dernier Rapport ({report_date})"):
                                if latest_other:
                                    st.write(f"**Autres:** {latest_other}")
                                if latest_note:
                                    st.write(f"**Notes:** {latest_note}")
                        
                        # Add metabolic monitoring section
                        st.markdown("---")
                        st.subheader("Surveillance Métabolique")
                        
                        # Check if metabolic data is available (simplified example)
                        if 'weight' in patient_data and 'glucose' in patient_data:
                            # This would be replaced with actual metabolic monitoring data
                            st.info("Données de surveillance métabolique disponibles dans le dossier médical.")
                            
                            # Create a sample table
                            metabolic_data = {
                                "Paramètre": ["Poids (kg)", "IMC", "Tour de taille (cm)", "Glycémie à jeun (mmol/L)", "HbA1c (%)", "Cholestérol total (mmol/L)", "LDL (mmol/L)", "HDL (mmol/L)", "Triglycérides (mmol/L)"],
                                "Valeur Baseline": ["72.5", "24.8", "86", "5.2", "5.4", "4.8", "2.6", "1.2", "1.8"],
                                "Dernière Valeur": ["78.2", "26.7", "92", "5.8", "5.7", "5.1", "2.9", "1.0", "2.3"],
                                "Changement": ["+5.7", "+1.9", "+6", "+0.6", "+0.3", "+0.3", "+0.3", "-0.2", "+0.5"],
                                "Date": ["2023-06-15", "2023-06-15", "2023-06-15", "2023-06-15", "2023-06-15", "2023-06-15", "2023-06-15", "2023-06-15", "2023-06-15"]
                            }
                            
                            metabolic_df = pd.DataFrame(metabolic_data)
                            st.dataframe(metabolic_df, hide_index=True, use_container_width=True)
                            
                            # Add reference ranges expandable section
                            with st.expander("Valeurs de référence"):
                                st.markdown("""
                                - **IMC normal:** 18.5-24.9 kg/m²
                                - **Tour de taille à risque:** >94 cm (H), >80 cm (F)
                                - **Glycémie à jeun normale:** 3.9-5.6 mmol/L
                                - **HbA1c normale:** <5.7%
                                - **Cholestérol total souhaitable:** <5.2 mmol/L
                                - **LDL souhaitable:** <3.4 mmol/L
                                - **HDL souhaitable:** >1.0 mmol/L (H), >1.3 mmol/L (F)
                                - **Triglycérides normaux:** <1.7 mmol/L
                                """)
                        else:
                            st.warning("Données de surveillance métabolique non disponibles.")
            else:
                st.info(f"ℹ️ Aucun rapport d'effets secondaires trouvé pour {patient_id}.")
        except Exception as e:
            st.error(f"Erreur chargement résumé effets secondaires: {e}")
            logging.exception(f"Error loading SE summary {patient_id}")

    # --- Tab 7: Nurse Notes History (no major changes needed) ---
    with tab_notes_history:
        st.header("📝 Historique Notes Infirmières")
        st.info("Affiche les notes et plans de soins précédents.")
        try:
            notes_history_df = get_nurse_inputs_history(patient_id)
            if notes_history_df.empty: st.info(f"ℹ️ Aucune note historique pour {patient_id}.")
            else:
                st.info(f"Affichage de {len(notes_history_df)} entrées.")
                display_columns = ['timestamp', 'goal_status', 'objectives', 'tasks', 'target_symptoms', 'planned_interventions', 'comments', 'created_by']
                display_columns = [col for col in display_columns if col in notes_history_df.columns]
                display_df_hist = notes_history_df[display_columns].copy()
                rename_map = { 'timestamp': 'Date/Heure', 'goal_status': 'Statut', 'objectives': 'Objectifs', 'tasks': 'Tâches','target_symptoms': 'Sympt. Cibles', 'planned_interventions': 'Interventions', 'comments': 'Comm.', 'created_by': 'Auteur' }
                display_df_hist.rename(columns={k: v for k, v in rename_map.items() if k in display_df_hist.columns}, inplace=True)
                if 'Date/Heure' in display_df_hist.columns: display_df_hist['Date/Heure'] = pd.to_datetime(display_df_hist['Date/Heure'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                for index, row in display_df_hist.iterrows():
                    exp_title = f"Entrée {row.get('Date/Heure', 'N/A')} (Statut: {row.get('Statut', 'N/A')})"
                    author = row.get('Auteur', None);
                    if pd.notna(author) and author: exp_title += f" - {author}"
                    with st.expander(exp_title):
                        st.markdown(f"**Statut:** {row.get('Statut', 'N/A')} | **Sympt Cibles:** {row.get('Sympt. Cibles', 'N/A')} | **Interv:** {row.get('Interventions', 'N/A')}")
                        st.markdown(f"**Objectifs:**\n{row.get('Objectifs', 'N/A')}"); st.markdown(f"**Tâches:**\n{row.get('Tâches', 'N/A')}"); st.markdown(f"**Comm:**\n{row.get('Comm.', 'N/A')}")
        except Exception as e: st.error(f"Erreur historique notes: {e}"); logging.exception(f"Error loading notes history {patient_id}")