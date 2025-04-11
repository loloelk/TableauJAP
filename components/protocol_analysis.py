# components/protocol_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # S'assurer que numpy est importé

def protocol_analysis_page():
    """Page pour analyser les interventions pharmacologiques"""
    st.header("💊 Analyse des Interventions Pharmacologiques")

    # --- Vérification des Données ---
    if 'final_data' not in st.session_state or st.session_state.final_data.empty:
        st.error("❌ Aucune donnée patient chargée. Impossible d'analyser les interventions.")
        return

    # Vérifier si les colonnes essentielles existent
    colonnes_requises = ['protocol', 'panss_pos_bl', 'panss_pos_fu']
    if not all(col in st.session_state.final_data.columns for col in colonnes_requises):
        st.error(f"❌ Colonnes requises manquantes dans les données: {', '.join(colonnes_requises)}. Vérifiez le fichier CSV.")
        return

    # Dictionnaire de traduction pour les protocoles médicamenteux
    traduction_protocoles = {
        'Long-acting Injectable': 'Injectable', 
        'Clozapine Pathway': 'Clozapine',
        'Standard Antipsychotic': 'Antipsychotique atypique',
        'Combined AP+MS': 'APA+Stabilisateur'
    }
    
    # Créer une copie des données pour éviter de modifier les données originales
    donnees_traitees = st.session_state.final_data.copy()
    
    # Traduire les noms des protocoles
    if 'protocol' in donnees_traitees.columns:
        donnees_traitees['protocol'] = donnees_traitees['protocol'].map(lambda x: traduction_protocoles.get(x, x))
    
    # Mettre à jour les données dans la session pour cette page
    st.session_state.donnees_protocoles = donnees_traitees

    tous_protocoles = sorted(donnees_traitees['protocol'].dropna().unique().tolist())
    if not tous_protocoles:
         st.warning("⚠️ Aucune information sur les médicaments trouvée dans les données.")
         return


    # --- Créer les Onglets ---
    onglet_dist, onglet_efficacite, onglet_comparaison, onglet_effets_sec = st.tabs([
        "👥 Distribution",
        "📈 Efficacité Moyenne",
        "🆚 Comparaison Détaillée",
        "⚠️ Effets Secondaires"
    ])

    # --- Onglet 1: Distribution ---
    with onglet_dist:
        st.subheader("Distribution des Patients par Traitement")

        nombre_protocoles = donnees_traitees['protocol'].value_counts().reset_index()
        nombre_protocoles.columns = ['Traitement', 'Nombre de Patients']

        fig_dist = px.bar(
            nombre_protocoles, x='Traitement', y='Nombre de Patients',
            color='Traitement', title="Répartition des Patients par Intervention Pharmacologique",
            text='Nombre de Patients'
        )
        fig_dist.update_traces(textposition='outside')
        st.plotly_chart(fig_dist, use_container_width=True)

        st.dataframe(nombre_protocoles, hide_index=True, use_container_width=True)

        if st.checkbox("Afficher en diagramme circulaire", key="dist_pie_cb"):
            fig_cercle = px.pie(
                nombre_protocoles, values='Nombre de Patients', names='Traitement',
                title="Distribution des Traitements Pharmacologiques"
            )
            st.plotly_chart(fig_cercle, use_container_width=True)
            
        # Ajouter la distribution des médicaments (pas seulement les protocoles)
        if 'medications' in donnees_traitees.columns:
            st.subheader("Médicaments Prescrits")
            
            # Extraire tous les noms de médicaments de la colonne medications
            tous_medicaments = []
            for meds in donnees_traitees['medications'].dropna():
                if isinstance(meds, str) and meds != "Aucun":
                    # Séparer par point-virgule et extraire juste le nom du médicament sans dosage
                    liste_meds = meds.split(';')
                    for med in liste_meds:
                        nom_med = med.strip().split(' ')[0]  # Obtenir juste le nom
                        tous_medicaments.append(nom_med)
            
            if tous_medicaments:
                nombre_meds = pd.Series(tous_medicaments).value_counts().reset_index()
                nombre_meds.columns = ['Médicament', 'Fréquence']
                
                fig_meds = px.bar(
                    nombre_meds.head(10), x='Médicament', y='Fréquence',
                    color='Médicament', title="Top 10 des Médicaments Prescrits",
                    text='Fréquence'
                )
                fig_meds.update_traces(textposition='outside')
                st.plotly_chart(fig_meds, use_container_width=True)

    # Préparer les données pour les onglets d'Efficacité et Comparaison (réduction des symptômes positifs PANSS)
    df_panss = donnees_traitees[['protocol', 'panss_pos_bl', 'panss_pos_fu']].copy()
    df_panss.dropna(subset=['panss_pos_bl', 'panss_pos_fu'], inplace=True) # Utiliser seulement les patients avec les deux scores

    if df_panss.empty:
         st.warning("⚠️ Aucune donnée PANSS complète (baseline et suivi) disponible pour l'analyse d'efficacité.")
         # Éviter les erreurs dans les onglets suivants si les données sont manquantes
         donnees_valides_pour_analyse = False
    else:
         donnees_valides_pour_analyse = True
         # Calculer l'amélioration (réduction des symptômes positifs PANSS)
         # Échelle PANSS positive: 7-49, plus élevé signifie pire, donc l'amélioration est une réduction
         df_panss['amelioration'] = df_panss['panss_pos_bl'] - df_panss['panss_pos_fu']
         df_panss['amelioration_pct'] = np.where(
             df_panss['panss_pos_bl'] > 7,  # Score PANSS positif minimum est 7
             (df_panss['amelioration'] / (df_panss['panss_pos_bl'] - 7) * 100),  # Normaliser pour tenir compte du minimum
             0  # Attribuer 0% d'amélioration si la baseline est au minimum
         )
         
         # Calculer la réponse cliniquement significative (30% de réduction du PANSS positif est standard)
         df_panss['repondeur'] = df_panss['amelioration_pct'] >= 30
         
         # Utiliser clinical_response et functional_response des données si disponibles
         if 'clinical_response' in donnees_traitees.columns:
             df_panss['reponse_clinique'] = donnees_traitees['clinical_response']
         else:
             df_panss['reponse_clinique'] = df_panss['repondeur']
             
         if 'functional_response' in donnees_traitees.columns:
             df_panss['reponse_fonctionnelle'] = donnees_traitees['functional_response']


    # --- Onglet 2: Efficacité ---
    with onglet_efficacite:
        st.subheader("Efficacité Moyenne des Traitements (Basée sur Symptômes Positifs PANSS)")

        if not donnees_valides_pour_analyse:
             st.warning("Données PANSS insuffisantes pour l'analyse.")
        else:
            # Regrouper par protocole
            metriques_protocole = df_panss.groupby('protocol').agg(
                 N=('protocol', 'size'),
                 Amelioration_Pts_Moyenne=('amelioration', 'mean'),
                 Amelioration_Pct_Moyenne=('amelioration_pct', 'mean'),
                 Taux_Reponse_Pct=('reponse_clinique', lambda x: x.mean() * 100),
            ).reset_index()
            
            # Ajouter la réponse fonctionnelle si disponible
            if 'reponse_fonctionnelle' in df_panss.columns:
                reponse_fonctionnelle = df_panss.groupby('protocol')['reponse_fonctionnelle'].mean() * 100
                metriques_protocole = metriques_protocole.merge(
                    reponse_fonctionnelle.reset_index().rename(columns={'reponse_fonctionnelle': 'Taux_Remission_Pct'}),
                    on='protocol'
                )
            else:
                metriques_protocole['Taux_Remission_Pct'] = np.nan

            # Formater les colonnes
            metriques_protocole['Amelioration_Pts_Moyenne'] = metriques_protocole['Amelioration_Pts_Moyenne'].round(1)
            metriques_protocole['Amelioration_Pct_Moyenne'] = metriques_protocole['Amelioration_Pct_Moyenne'].round(1)
            metriques_protocole['Taux_Reponse_Pct'] = metriques_protocole['Taux_Reponse_Pct'].round(1)
            if 'Taux_Remission_Pct' in metriques_protocole.columns:
                metriques_protocole['Taux_Remission_Pct'] = metriques_protocole['Taux_Remission_Pct'].round(1)

            metriques_protocole.rename(columns={
                 'protocol': 'Traitement',
                 'N': 'Nb Patients (PANSS Complet)',
                 'Amelioration_Pts_Moyenne': 'Amélioration Moyenne (Points)',
                 'Amelioration_Pct_Moyenne': 'Amélioration Moyenne (%)',
                 'Taux_Reponse_Pct': 'Taux Réponse Clinique (%)',
                 'Taux_Remission_Pct': 'Taux Réponse Fonctionnelle (%)'
            }, inplace=True)

            st.dataframe(metriques_protocole, hide_index=True, use_container_width=True)

            # Graphique à barres pour le pourcentage d'amélioration
            fig_amelio = px.bar(
                 metriques_protocole, x='Traitement', y='Amélioration Moyenne (%)',
                 color='Traitement', title="Pourcentage d'Amélioration PANSS Moyen par Traitement",
                 text='Amélioration Moyenne (%)'
            )
            fig_amelio.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_amelio, use_container_width=True)

            # Graphique à barres groupées pour les taux de réponse et de rémission
            colonnes_reponse = ['Taux Réponse Clinique (%)']
            if 'Taux Réponse Fonctionnelle (%)' in metriques_protocole.columns:
                colonnes_reponse.append('Taux Réponse Fonctionnelle (%)')
                
            donnees_taux = pd.melt(
                metriques_protocole,
                id_vars=['Traitement'],
                value_vars=colonnes_reponse,
                var_name='Mesure', value_name='Pourcentage'
            )
            fig_taux = px.bar(
                donnees_taux, x='Traitement', y='Pourcentage', color='Mesure',
                barmode='group', title="Taux de Réponse par Traitement",
                text='Pourcentage'
            )
            fig_taux.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_taux.update_layout(yaxis_title="Pourcentage (%)")
            st.plotly_chart(fig_taux, use_container_width=True)
            
            # Ajouter l'analyse CGI (Impression Clinique Globale) si disponible
            if 'cgi_s_bl' in donnees_traitees.columns and 'cgi_s_fu' in donnees_traitees.columns:
                st.subheader("Amélioration Clinique Globale (CGI)")
                
                df_cgi = donnees_traitees[['protocol', 'cgi_s_bl', 'cgi_s_fu']].copy()
                df_cgi.dropna(subset=['cgi_s_bl', 'cgi_s_fu'], inplace=True)
                
                # Calculer l'amélioration CGI (réduction de la sévérité)
                df_cgi['amelioration_cgi'] = df_cgi['cgi_s_bl'] - df_cgi['cgi_s_fu']
                
                # Amélioration moyenne par protocole
                amelioration_cgi = df_cgi.groupby('protocol')['amelioration_cgi'].mean().reset_index()
                amelioration_cgi.columns = ['Traitement', 'Amélioration CGI Moyenne']
                amelioration_cgi['Amélioration CGI Moyenne'] = amelioration_cgi['Amélioration CGI Moyenne'].round(2)
                
                # Graphique d'amélioration CGI
                fig_cgi = px.bar(
                    amelioration_cgi, 
                    x='Traitement', 
                    y='Amélioration CGI Moyenne',
                    color='Traitement', 
                    title="Amélioration de l'Impression Clinique Globale par Traitement",
                    text='Amélioration CGI Moyenne'
                )
                fig_cgi.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig_cgi, use_container_width=True)
                
                # Expliquer l'échelle CGI
                with st.expander("Explication de l'échelle CGI"):
                    st.markdown("""
                    **Échelle d'Impression Clinique Globale (CGI)**
                    
                    * **CGI-S (Sévérité)**: 1=normal, 2=limite, 3=légèrement malade, 4=modérément malade, 5=manifestement malade, 6=gravement malade, 7=extrêmement malade
                    * **Amélioration CGI**: Une valeur positive indique une réduction de la sévérité de la maladie
                    """)


    # --- Onglet 3: Comparaison Détaillée ---
    with onglet_comparaison:
        st.subheader("Comparaison Détaillée des Traitements")

        if not donnees_valides_pour_analyse:
             st.warning("Données PANSS insuffisantes pour l'analyse détaillée.")
        else:
            # Permettre à l'utilisateur de sélectionner les protocoles à comparer
            protocoles_selectionnes = st.multiselect(
                "Sélectionner les traitements à comparer:",
                options=tous_protocoles,
                default=tous_protocoles[:2] if len(tous_protocoles) >= 2 else tous_protocoles,
                key="protocol_compare_multiselect"
            )

            if not protocoles_selectionnes:
                st.warning("Veuillez sélectionner au moins un traitement.")
            else:
                # Filtrer les données pour les protocoles sélectionnés
                df_comparaison = df_panss[df_panss['protocol'].isin(protocoles_selectionnes)].copy()

                if df_comparaison.empty:
                     st.warning("Aucune donnée pour les traitements sélectionnés.")
                else:
                    # Permettre à l'utilisateur de choisir entre différentes métriques
                    options_metriques = {
                        "amelioration_pct": "Pourcentage d'amélioration des symptômes positifs (PANSS)",
                        "amelioration": "Réduction des points de symptômes positifs (PANSS)"
                    }
                    
                    if 'amelioration_cgi' in df_comparaison.columns:
                        options_metriques["amelioration_cgi"] = "Amélioration de l'impression clinique globale (CGI)"
                        
                    metrique_selectionnee = st.selectbox(
                        "Sélectionner la métrique à comparer:",
                        options=list(options_metriques.keys()),
                        format_func=lambda x: options_metriques[x],
                        key="metric_selector"
                    )
                    
                    label_metrique = options_metriques[metrique_selectionnee].split('(')[0].strip()
                    
                    st.markdown(f"#### Comparaison basée sur {label_metrique}")

                    col_boite, col_points = st.columns(2)
                    with col_boite:
                         # Boîte à moustaches pour la distribution
                         st.markdown("**Distribution des Améliorations**")
                         fig_boite = px.box(
                              df_comparaison, x='protocol', y=metrique_selectionnee,
                              color='protocol', title="Distribution",
                              labels={'protocol': 'Traitement', metrique_selectionnee: label_metrique},
                              points="all"  # Afficher tous les points individuels
                         )
                         st.plotly_chart(fig_boite, use_container_width=True)
                    with col_points:
                         # Graphique en points (vue alternative des points individuels)
                         st.markdown("**Points Individuels**")
                         fig_points = px.strip(
                              df_comparaison, x='protocol', y=metrique_selectionnee,
                              color='protocol', title="Points Individuels",
                              labels={'protocol': 'Traitement', metrique_selectionnee: label_metrique}
                         )
                         st.plotly_chart(fig_points, use_container_width=True)


                    # Résumé statistique
                    st.markdown("---")
                    st.subheader(f"Résumé Statistique - {label_metrique}")
                    df_stats = df_comparaison.groupby('protocol')[metrique_selectionnee].describe().reset_index()
                    # Renommer les colonnes pour plus de clarté
                    df_stats.rename(columns={
                         'protocol':'Traitement', 'count':'N', 'mean':'Moyenne', 'std':'Écart-Type',
                         'min':'Min', '25%':'25ème Perc.', '50%':'Médiane', '75%':'75ème Perc.', 'max':'Max'
                    }, inplace=True)
                    # Formater les colonnes numériques
                    colonnes_num = df_stats.columns.drop(['Traitement', 'N'])
                    df_stats[colonnes_num] = df_stats[colonnes_num].round(1)
                    st.dataframe(df_stats, hide_index=True, use_container_width=True)

                    # --- Comparaison de Différence Moyenne ---
                    st.markdown("---")
                    st.subheader(f"Comparaison Directe des Moyennes de {label_metrique}")

                    if len(protocoles_selectionnes) < 2:
                        st.info("Sélectionnez au moins deux traitements pour voir une comparaison directe.")
                    elif len(protocoles_selectionnes) == 2:
                         # Comparaison directe pour deux protocoles
                         proto1 = protocoles_selectionnes[0]
                         proto2 = protocoles_selectionnes[1]
                         moyenne1 = df_stats.loc[df_stats['Traitement'] == proto1, 'Moyenne'].iloc[0]
                         moyenne2 = df_stats.loc[df_stats['Traitement'] == proto2, 'Moyenne'].iloc[0]
                         diff = moyenne1 - moyenne2

                         st.metric(
                              label=f"Différence Moyenne ({proto1} vs {proto2})",
                              value=f"{diff:.1f}",
                              help=f"Une valeur positive signifie que {proto1} a une amélioration moyenne supérieure à {proto2} dans cet échantillon."
                         )
                    else:
                         # Comparaison matricielle pour plus de deux protocoles
                         st.write("Différences moyennes entre les traitements (Ligne - Colonne):")
                         # Table pivot pour faciliter la recherche
                         pivot_moyennes = df_stats.set_index('Traitement')['Moyenne']
                         # Créer une matrice vide
                         matrice_diff = pd.DataFrame(index=protocoles_selectionnes, columns=protocoles_selectionnes, dtype=float)

                         for p1 in protocoles_selectionnes:
                              for p2 in protocoles_selectionnes:
                                   if p1 != p2:
                                        moyenne1 = pivot_moyennes.get(p1, np.nan)
                                        moyenne2 = pivot_moyennes.get(p2, np.nan)
                                        if not pd.isna(moyenne1) and not pd.isna(moyenne2):
                                             matrice_diff.loc[p1, p2] = round(moyenne1 - moyenne2, 1)
                                        else:
                                             matrice_diff.loc[p1, p2] = np.nan # Marquer comme NaN si moyenne manquante
                                   else:
                                        matrice_diff.loc[p1, p2] = 0.0 # Différence avec soi-même est 0

                         # Afficher la matrice (en utilisant le dataframe de Streamlit pour un meilleur formatage)
                         st.dataframe(matrice_diff.style.format("{:.1f}", na_rep="-").highlight_null(color='lightgray'))
                         st.caption("Les valeurs positives indiquent que le traitement en ligne a une meilleure amélioration moyenne que le traitement en colonne.")

                    st.info("ℹ️ Note: Ces différences sont basées sur les moyennes de cet échantillon. Une analyse statistique plus rigoureuse (tests t, ANOVA) serait nécessaire pour déterminer la significativité statistique dans un contexte clinique.")

    
    # --- Onglet 4: Analyse des Effets Secondaires ---
    with onglet_effets_sec:
        st.subheader("Analyse des Effets Secondaires par Traitement")
        
        # Check if we have side effects data
        if 'side_effects' not in st.session_state:
            # Try to load side effects data from database/CSV
            from services.nurse_service import get_side_effects_history
            
            # Combine side effects data for all patients
            all_side_effects = pd.DataFrame()
            
            for patient_id in st.session_state.final_data['ID'].unique():
                patient_effects = get_side_effects_history(patient_id)
                if not patient_effects.empty:
                    all_side_effects = pd.concat([all_side_effects, patient_effects])
            
            if all_side_effects.empty:
                st.warning("Aucune donnée d'effets secondaires trouvée pour l'analyse.")
                with st.expander("Visualisation des Profils d'Effets Secondaires Connus"):
                    # Generate simulated data for visualization based on known side effect profiles
                    st.markdown("""
                    ### Profils d'Effets Secondaires Typiques par Classe de Médicaments
                    
                    Les données réelles d'effets secondaires ne sont pas disponibles. Voici une visualisation des profils 
                    typiques d'effets secondaires pour différentes classes d'antipsychotiques basée sur la littérature.
                    """)
                    
                    # Create simulated data
                    drug_classes = [
                        "Antipsychotiques Typiques", 
                        "Antipsychotiques Atypiques",
                        "Clozapine", 
                        "Antipsychotiques Injectables"
                    ]
                    
                    side_effects = [
                        "EPS", "Akathisie", "Prise de poids", 
                        "Effets Métaboliques", "Sédation", "Dysfonction Sexuelle"
                    ]
                    
                    # Typical side effect profiles (scale 0-10)
                    profiles = {
                        "Antipsychotiques Typiques": [8, 7, 3, 2, 5, 4],
                        "Antipsychotiques Atypiques": [3, 4, 7, 6, 5, 5],
                        "Clozapine": [1, 2, 9, 8, 8, 6],
                        "Antipsychotiques Injectables": [4, 5, 5, 4, 3, 3]
                    }
                    
                    # Create radar chart with simulated data
                    fig = go.Figure()
                    
                    for drug_class in drug_classes:
                        values = profiles[drug_class]
                        # Add the first value at the end to close the loop
                        values_closed = values + [values[0]]
                        side_effects_closed = side_effects + [side_effects[0]]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values_closed,
                            theta=side_effects_closed,
                            fill='toself',
                            name=drug_class
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10]
                            )
                        ),
                        title="Profils d'Effets Secondaires Typiques (Données Simulées)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Note**: Cette visualisation est basée sur des profils généraux et peut ne pas refléter 
                    l'expérience individuelle des patients. Les effets secondaires varient considérablement 
                    d'un patient à l'autre.
                    """)
                return
            
            # Store in session state for future use
            st.session_state.side_effects = all_side_effects
        
        # Get patient treatment info
        patient_treatments = st.session_state.final_data[['ID', 'protocol']].dropna().rename(
            columns={'ID': 'patient_id', 'protocol': 'treatment'})
        
        # Merge side effects with treatment info
        side_effects_with_treatment = pd.merge(
            st.session_state.side_effects,
            patient_treatments,
            on='patient_id',
            how='inner'
        )
        
        if side_effects_with_treatment.empty:
            st.warning("Impossible de joindre les données d'effets secondaires avec les informations de traitement.")
            return
        
        # Define side effect columns - check which ones exist in the data
        possible_effect_cols = ['eps', 'akathisia', 'weight_gain', 'metabolic_changes', 
                               'sedation', 'sexual_dysfunction', 'headache', 'nausea', 
                               'scalp_discomfort', 'dizziness']
        
        effect_cols = [col for col in possible_effect_cols if col in side_effects_with_treatment.columns]
        
        if not effect_cols:
            st.warning("Aucune colonne d'effet secondaire reconnue trouvée dans les données.")
            return
        
        # French translations for side effects
        effect_labels = {
            'eps': 'Sympt. extrapyramidaux',
            'akathisia': 'Akathisie',
            'weight_gain': 'Prise de poids',
            'metabolic_changes': 'Chgt. métaboliques',
            'sedation': 'Sédation',
            'sexual_dysfunction': 'Dysf. sexuelle',
            'headache': 'Maux de tête',
            'nausea': 'Nausée',
            'scalp_discomfort': 'Inconfort du cuir chevelu',
            'dizziness': 'Vertiges'
        }
        
        # Calculate average side effect severity by treatment
        treatment_effects = side_effects_with_treatment.groupby('treatment')[effect_cols].mean().reset_index()
        
        # Create a radar chart for each treatment
        st.subheader("Profil des Effets Secondaires par Traitement")
        
        # Select treatments to visualize
        treatments_to_show = st.multiselect(
            "Sélectionner les traitements à comparer:",
            options=treatment_effects['treatment'].unique(),
            default=treatment_effects['treatment'].unique()[:3] if len(treatment_effects) >= 3 else treatment_effects['treatment'].unique()
        )
        
        if treatments_to_show:
            filtered_effects = treatment_effects[treatment_effects['treatment'].isin(treatments_to_show)]
            
            # Create radar chart
            fig = go.Figure()
            
            for treatment in treatments_to_show:
                treatment_data = filtered_effects[filtered_effects['treatment'] == treatment]
                if not treatment_data.empty:
                    values = treatment_data[effect_cols].values.flatten().tolist()
                    # Add the first value at the end to close the loop
                    values = values + [values[0]]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=[effect_labels.get(col, col) for col in effect_cols] + [effect_labels.get(effect_cols[0], effect_cols[0])],
                        fill='toself',
                        name=treatment
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                title="Comparaison des Profils d'Effets Secondaires"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display a table with the average side effect values
            st.subheader("Sévérité Moyenne des Effets Secondaires")
            
            # Prepare data for display
            display_df = filtered_effects.copy()
            
            # Map column names to French labels if they exist
            display_df = display_df.rename(columns={col: effect_labels.get(col, col) for col in effect_cols})
            display_df = display_df.rename(columns={'treatment': 'Traitement'})
            
            # Round to 1 decimal place
            numeric_cols = [effect_labels.get(col, col) for col in effect_cols]
            display_df[numeric_cols] = display_df[numeric_cols].round(1)
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # Bar chart comparison of a specific side effect
            st.subheader("Comparaison d'un Effet Secondaire Spécifique")
            
            selected_effect_label = st.selectbox(
                "Sélectionner un effet secondaire à comparer:",
                options=[effect_labels.get(col, col) for col in effect_cols]
            )
            
            # Get the original column name
            selected_col = next((col for col in effect_cols if effect_labels.get(col, col) == selected_effect_label), effect_cols[0])
            
            # Create bar chart
            fig_bar = px.bar(
                filtered_effects, 
                x='treatment', 
                y=selected_col,
                color='treatment',
                title=f"Comparaison de {selected_effect_label} par Traitement",
                labels={'treatment': 'Traitement', selected_col: f'Sévérité Moyenne (0-10)'},
                text_auto='.1f'
            )
            
            fig_bar.update_layout(yaxis_range=[0, 10])
            st.plotly_chart(fig_bar, use_container_width=True)
        
        else:
            st.warning("Veuillez sélectionner au moins un traitement pour visualiser les effets secondaires.")