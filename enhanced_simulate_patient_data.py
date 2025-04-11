# enhanced_simulate_patient_data.py

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
import math
import logging
import sys

# --- Import functions to save data directly to DB ---
try:
    from services.nurse_service import save_nurse_inputs, save_side_effect_report, initialize_database
    DB_INTERACTION_ENABLED = True
except ImportError as e:
    print(f"WARNING: Could not import from services.nurse_service: {e}. Will save to CSV instead (Nurse/Side Effect data won't populate DB).")
    DB_INTERACTION_ENABLED = False
    def save_nurse_inputs(*args, **kwargs): pass
    def save_side_effect_report(*args, **kwargs): pass
    def initialize_database(): pass


# --- Configuration Parameters ---
NUM_PATIENTS = 50
PROTOCOLS = ['Standard Antipsychotic', 'Clozapine Pathway', 'Long-acting Injectable', 'Combined AP+MS']
START_DATE = datetime(2024, 1, 1)
SIMULATION_DURATION_DAYS = 90  # Longer monitoring period for FEP

# Protocol response probabilities (BASE rates)
PROTOCOL_RESPONSE_RATES = {
    'Standard Antipsychotic': 0.65, 
    'Clozapine Pathway': 0.75, 
    'Long-acting Injectable': 0.70, 
    'Combined AP+MS': 0.60
}

# Modified: No longer using remission rates but functional response (CGI) probabilities
PROTOCOL_FUNCTIONAL_RESPONSE_RATES = {
    'Standard Antipsychotic': 0.40, 
    'Clozapine Pathway': 0.45, 
    'Long-acting Injectable': 0.55, 
    'Combined AP+MS': 0.35
}

# Diagnosis categories
DIAGNOSES = ["First Episode Psychosis", "Psychosis NOS", "Schizophrenia", "Bipolar with Psychotic Features", "Schizoaffective"]
DIAGNOSIS_WEIGHTS = [0.35, 0.25, 0.15, 0.15, 0.10]  # First episode and NOS more common in this clinic

# EMA Simulation Parameters
EMA_MISSING_DAY_PROB = 0.15  # Higher missing rate for psychosis patients
EMA_MISSING_ENTRY_PROB = 0.20
EMA_ENTRIES_PER_DAY_WEIGHTS = [0.4, 0.5, 0.1]  # Less likely to complete all entries

# Side Effect Simulation Parameters
SIDE_EFFECT_PROB_INITIAL = 0.6  # Higher initial side effects for antipsychotics
SIDE_EFFECT_PROB_LATER = 0.4
SIDE_EFFECT_DECAY_DAY = 30  # Longer adaptation period

# Removed: BFI Simulation Parameters

# Medication categories for psychosis
MEDICATION_CATEGORIES = {
    'Atypical_Primary': ['Olanzapine', 'Risperidone', 'Aripiprazole', 'Quetiapine', 'Ziprasidone'],
    'Atypical_Secondary': ['Clozapine', 'Lurasidone', 'Paliperidone', 'Asenapine', 'Brexpiprazole'],
    'LAI': ['Aripiprazole Maintena', 'Risperdal Consta', 'Invega Sustenna', 'Abilify Mycite', 'Zyprexa Relprevv'],
    'Mood_Stabilizers': ['Lithium', 'Valproate', 'Lamotrigine', 'Carbamazepine', 'Oxcarbazepine'],
    'Other': ['Haloperidol', 'Fluphenazine', 'Perphenazine', 'Loxapine', 'Chlorpromazine']
}

# Medication dosage ranges (min, max, step)
MEDICATION_DOSAGES = {
    'Olanzapine': (5, 20, 2.5),
    'Risperidone': (1, 6, 0.5),
    'Aripiprazole': (5, 20, 2.5),
    'Quetiapine': (100, 800, 50),
    'Ziprasidone': (40, 160, 20),
    'Clozapine': (50, 350, 25),
    'Lurasidone': (40, 120, 20),
    'Paliperidone': (3, 12, 3),
    'Asenapine': (5, 20, 5),
    'Brexpiprazole': (1, 4, 0.5),
    'Aripiprazole Maintena': (300, 400, 100),
    'Risperdal Consta': (25, 50, 12.5),
    'Invega Sustenna': (78, 234, 39),
    'Abilify Mycite': (10, 30, 5),
    'Zyprexa Relprevv': (210, 405, 105),
    'Lithium': (600, 1200, 150),
    'Valproate': (500, 1500, 250),
    'Lamotrigine': (50, 200, 25),
    'Carbamazepine': (400, 1200, 200),
    'Oxcarbazepine': (600, 1200, 300),
    'Haloperidol': (2, 15, 1),
    'Fluphenazine': (1, 10, 1),
    'Perphenazine': (4, 32, 4),
    'Loxapine': (10, 100, 10),
    'Chlorpromazine': (50, 400, 50)
}

# Medication units - adding for clarity
MEDICATION_UNITS = {
    'Aripiprazole Maintena': 'mg/month',
    'Risperdal Consta': 'mg/2weeks',
    'Invega Sustenna': 'mg/month',
    'Zyprexa Relprevv': 'mg/2weeks',
    'Abilify Mycite': 'mg',
    'Lithium': 'mg',
    'Valproate': 'mg',
    'Lamotrigine': 'mg',
    'Carbamazepine': 'mg',
    'Oxcarbazepine': 'mg',
}  # All others default to 'mg'

# PANSS Simulation Parameters (7-point scale for 30 items)
PANSS_POSITIVE_ITEMS = ['P1_Delusions', 'P2_Conceptual_Disorganization', 'P3_Hallucinations', 
                       'P4_Excitement', 'P5_Grandiosity', 'P6_Suspiciousness', 'P7_Hostility']
PANSS_NEGATIVE_ITEMS = ['N1_Blunted_Affect', 'N2_Emotional_Withdrawal', 'N3_Poor_Rapport',
                       'N4_Passive_Withdrawal', 'N5_Abstract_Thinking', 'N6_Conversation_Flow', 'N7_Stereotyped_Thinking']
PANSS_GENERAL_ITEMS = ['G1_Somatic_Concern', 'G2_Anxiety', 'G3_Guilt', 'G4_Tension', 'G5_Mannerisms', 
                     'G6_Depression', 'G7_Motor_Retardation', 'G8_Uncooperativeness', 'G9_Unusual_Thought', 
                     'G10_Disorientation', 'G11_Poor_Attention', 'G12_Insight', 'G13_Volition', 
                     'G14_Impulse_Control', 'G15_Preoccupation', 'G16_Active_Avoidance']

# CGI Scale (Clinical Global Impression)
# CGI-S: 1=normal, 2=borderline, 3=mild, 4=moderate, 5=marked, 6=severe, 7=extreme
# CGI-I: 1=very much improved, 2=much improved, 3=minimally improved, 4=no change, 5=minimally worse, 6=much worse, 7=very much worse

# Calgary Depression Scale - 9 items, 0-3 scale
CALGARY_ITEMS = ['Depression', 'Hopelessness', 'Self_Depreciation', 'Guilty_Ideas', 
                'Pathological_Guilt', 'Morning_Depression', 'Early_Wakening', 
                'Suicide', 'Observed_Depression']

# PSYRATS - Auditory Hallucinations Scale (11 items, 0-4 scale)
PSYRATS_AH_ITEMS = ['Frequency', 'Duration', 'Location', 'Loudness', 'Beliefs_Origin', 
                   'Amount_Negative', 'Negative_Content', 'Distress', 'Disruption', 
                   'Controllability']

# PSYRATS - Delusions Scale (6 items, 0-4 scale)
PSYRATS_DEL_ITEMS = ['Preoccupation', 'Duration', 'Conviction', 'Distress', 
                    'Disruption', 'Insight']

# Side Effects for antipsychotics
SIDE_EFFECT_TYPES = {
    'EPS': ['Tremor', 'Rigidity', 'Bradykinesia', 'Dystonia'],
    'Akathisia': ['Inner_Restlessness', 'Inability_To_Keep_Still'],
    'Metabolic': ['Weight_Gain', 'Glucose_Elevation', 'Cholesterol_Elevation', 'Triglyceride_Elevation'],
    'Other': ['Sedation', 'Sexual_Dysfunction', 'Hyperprolactinemia', 'QTc_Prolongation', 'Dry_Mouth', 'Constipation']
}

# Create data directory if needed
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Configure basic logging for the script
log_file = os.path.join('logs', f'simulation_{datetime.now():%Y-%m-%d_%H%M%S}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)

# --- Helper Functions ---

def distribute_phq9_score(total_score):
    """Distribute a total PHQ-9 score into 9 item scores realistically"""
    # (Implementation remains the same as before)
    if total_score <= 0: return [0] * 9
    weights=[1.5, 1.5, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 0.5]; norm_w=[w/sum(weights) for w in weights]
    raw=[total_score*w for w in norm_w]; items=[min(3,max(0,round(v))) for v in raw]
    curr=sum(items)
    while curr<total_score and max(items)<3:
        el=[i for i,s in enumerate(items) if s<3];
        if not el: break
        idx=random.choices(el,[norm_w[i] for i in el])[0]; items[idx]+=1; curr+=1
    while curr>total_score and min(items)>0:
        el=[i for i,s in enumerate(items) if s>0];
        if not el: break
        inv_w=[1/norm_w[i] if norm_w[i]>0 else float('inf') for i in el]; sum_inv=sum(w for w in inv_w if w!=float('inf'))
        if sum_inv==0: break
        inv_n=[w/sum_inv if w!=float('inf') else 0 for w in inv_w]; idx=random.choices(el,inv_n)[0]; items[idx]-=1; curr-=1
    return items

def distribute_calgary_score(total_score):
    """Distribute a total Calgary Depression Scale score into 9 item scores realistically"""
    if total_score <= 0: 
        return [0] * 9
    
    # Weights reflecting typical item distributions
    weights = [1.5, 1.2, 1.0, 0.8, 0.7, 1.0, 0.8, 0.5, 1.2]
    norm_w = [w/sum(weights) for w in weights]
    
    # Initial raw distribution
    raw = [total_score * w for w in norm_w]
    items = [min(3, max(0, round(v))) for v in raw]
    
    # Adjust to match total
    curr = sum(items)
    
    # Increase items if needed
    while curr < total_score and max(items) < 3:
        el = [i for i, s in enumerate(items) if s < 3]
        if not el: break
        idx = random.choices(el, [norm_w[i] for i in el])[0]
        items[idx] += 1
        curr += 1
    
    # Decrease items if needed
    while curr > total_score and min(items) > 0:
        el = [i for i, s in enumerate(items) if s > 0]
        if not el: break
        # Fix the error here - using i as the index variable instead of w
        inv_w = [1/norm_w[i] if norm_w[i] > 0 else float('inf') for i in el]
        sum_inv = sum(w for w in inv_w if w != float('inf'))
        if sum_inv == 0: break
        inv_n = [w/sum_inv if w != float('inf') else 0 for w in inv_w]
        idx = random.choices(el, inv_n)[0]
        items[idx] -= 1
        curr -= 1
    
    return items

def distribute_panss_scores(total_positive, total_negative, total_general):
    """Distribute PANSS subscale scores to individual items"""
    positive_items = []
    negative_items = []
    general_items = []
    
    # Positive symptoms distribution (7 items, each 1-7 scale)
    # Minimum possible score is 7, so adjust total accordingly
    adjusted_pos = max(0, total_positive - 7)
    if adjusted_pos <= 0:
        positive_items = [1] * 7  # Minimum score
    else:
        # Weights reflecting typical distribution
        pos_weights = [1.5, 1.2, 1.4, 0.8, 0.7, 1.3, 0.8]
        pos_norm_w = [w/sum(pos_weights) for w in pos_weights]
        
        # Initial raw distribution (adding 1 as minimum for each item)
        pos_raw = [1 + (adjusted_pos * w) for w in pos_norm_w]
        positive_items = [min(7, max(1, round(v))) for v in pos_raw]
        
        # Adjust to match total
        curr_pos = sum(positive_items)
        
        # Adjust items to match total
        while curr_pos < total_positive and max(positive_items) < 7:
            el = [i for i, s in enumerate(positive_items) if s < 7]
            if not el: break
            idx = random.choices(el, [pos_norm_w[i] for i in el])[0]
            positive_items[idx] += 1
            curr_pos += 1
        
        while curr_pos > total_positive and max(positive_items) > 1:
            el = [i for i, s in enumerate(positive_items) if s > 1]
            if not el: break
            idx = random.choice(el)
            positive_items[idx] -= 1
            curr_pos -= 1
    
    # Negative symptoms distribution (similar approach)
    adjusted_neg = max(0, total_negative - 7)
    if adjusted_neg <= 0:
        negative_items = [1] * 7
    else:
        neg_weights = [1.2, 1.3, 1.0, 1.2, 0.9, 0.8, 0.7]
        neg_norm_w = [w/sum(neg_weights) for w in neg_weights]
        
        neg_raw = [1 + (adjusted_neg * w) for w in neg_norm_w]
        negative_items = [min(7, max(1, round(v))) for v in neg_raw]
        
        curr_neg = sum(negative_items)
        
        while curr_neg < total_negative and max(negative_items) < 7:
            el = [i for i, s in enumerate(negative_items) if s < 7]
            if not el: break
            idx = random.choices(el, [neg_norm_w[i] for i in el])[0]
            negative_items[idx] += 1
            curr_neg += 1
        
        while curr_neg > total_negative and max(negative_items) > 1:
            el = [i for i, s in enumerate(negative_items) if s > 1]
            if not el: break
            idx = random.choice(el)
            negative_items[idx] -= 1
            curr_neg -= 1
    
    # General psychopathology symptoms (16 items)
    adjusted_gen = max(0, total_general - 16)
    if adjusted_gen <= 0:
        general_items = [1] * 16
    else:
        # Less variance in general symptoms
        gen_weights = [1.0] * 16
        gen_norm_w = [w/sum(gen_weights) for w in gen_weights]
        
        gen_raw = [1 + (adjusted_gen * w) for w in gen_norm_w]
        general_items = [min(7, max(1, round(v))) for v in gen_raw]
        
        curr_gen = sum(general_items)
        
        while curr_gen < total_general and max(general_items) < 7:
            el = [i for i, s in enumerate(general_items) if s < 7]
            if not el: break
            idx = random.choices(el, [gen_norm_w[i] for i in el])[0]
            general_items[idx] += 1
            curr_gen += 1
        
        while curr_gen > total_general and max(general_items) > 1:
            el = [i for i, s in enumerate(general_items) if s > 1]
            if not el: break
            idx = random.choice(el)
            general_items[idx] -= 1
            curr_gen -= 1
    
    return positive_items, negative_items, general_items

def distribute_psyrats_scores(total_ah, total_del):
    """Distribute PSYRATS subscale scores to individual items"""
    # Auditory hallucinations (11 items, each 0-4 scale)
    ah_items = []
    if total_ah <= 0:
        ah_items = [0] * 10
    else:
        # Higher weights for frequency, distress and controllability
        ah_weights = [1.5, 1.2, 1.0, 1.0, 1.1, 1.3, 1.4, 1.5, 1.2, 1.5]
        ah_norm_w = [w/sum(ah_weights) for w in ah_weights]
        
        # Initial raw distribution
        ah_raw = [total_ah * w for w in ah_norm_w]
        ah_items = [min(4, max(0, round(v))) for v in ah_raw]
        
        # Adjust to match total
        curr_ah = sum(ah_items)
        
        while curr_ah < total_ah and max(ah_items) < 4:
            el = [i for i, s in enumerate(ah_items) if s < 4]
            if not el: break
            idx = random.choices(el, [ah_norm_w[i] for i in el])[0]
            ah_items[idx] += 1
            curr_ah += 1
        
        while curr_ah > total_ah and min(ah_items) > 0:
            el = [i for i, s in enumerate(ah_items) if s > 0]
            if not el: break
            idx = random.choice(el)
            ah_items[idx] -= 1
            curr_ah -= 1
    
    # Delusions (6 items, each 0-4 scale)
    del_items = []
    if total_del <= 0:
        del_items = [0] * 6
    else:
        # Higher weights for conviction and preoccupation
        del_weights = [1.5, 1.2, 1.6, 1.4, 1.0, 1.3]
        del_norm_w = [w/sum(del_weights) for w in del_weights]
        
        # Initial raw distribution
        del_raw = [total_del * w for w in del_norm_w]
        del_items = [min(4, max(0, round(v))) for v in del_raw]
        
        # Adjust to match total
        curr_del = sum(del_items)
        
        while curr_del < total_del and max(del_items) < 4:
            el = [i for i, s in enumerate(del_items) if s < 4]
            if not el: break
            idx = random.choices(el, [del_norm_w[i] for i in el])[0]
            del_items[idx] += 1
            curr_del += 1
        
        while curr_del > total_del and min(del_items) > 0:
            el = [i for i, s in enumerate(del_items) if s > 0]
            if not el: break
            idx = random.choice(el)
            del_items[idx] -= 1
            curr_del -= 1
    
    return ah_items, del_items

def generate_medication_data(patient, will_respond):
    """Generate psychiatric medication data for a patient with psychosis"""
    medications = []
    
    # Determine protocol-appropriate medications
    protocol = patient.get('protocol', 'Standard Antipsychotic')
    diagnosis = patient.get('diagnosis', 'First Episode Psychosis')
    
    # Special case for first 5 patients to ensure variety in demo
    patient_id = patient['ID']
    if patient_id == 'P001':
        # First episode on low-dose risperidone
        med = {'name': 'Risperidone', 'category': 'Atypical_Primary', 'dosage': 2, 'units': 'mg'}
        medications.append(med)
    elif patient_id == 'P002':
        # Treatment resistant on clozapine
        med = {'name': 'Clozapine', 'category': 'Atypical_Secondary', 'dosage': 250, 'units': 'mg'}
        medications.append(med)
    elif patient_id == 'P003':
        # Schizoaffective with mood stabilizer combination
        med1 = {'name': 'Olanzapine', 'category': 'Atypical_Primary', 'dosage': 10, 'units': 'mg'}
        med2 = {'name': 'Lithium', 'category': 'Mood_Stabilizers', 'dosage': 900, 'units': 'mg'}
        medications.extend([med1, med2])
    elif patient_id == 'P004':
        # Adherence issues on LAI
        med = {'name': 'Aripiprazole Maintena', 'category': 'LAI', 'dosage': 400, 'units': 'mg/month'}
        medications.append(med)
    elif patient_id == 'P005':
        # Bipolar with psychotic features
        med1 = {'name': 'Quetiapine', 'category': 'Atypical_Primary', 'dosage': 300, 'units': 'mg'}
        med2 = {'name': 'Valproate', 'category': 'Mood_Stabilizers', 'dosage': 1000, 'units': 'mg'}
        medications.extend([med1, med2])
    else:
        # Medication assignment based on protocol
        if protocol == 'Standard Antipsychotic':
            # Single atypical antipsychotic
            primary_category = 'Atypical_Primary'
            primary_med = random.choice(MEDICATION_CATEGORIES[primary_category])
            min_dose, max_dose, step = MEDICATION_DOSAGES[primary_med]
            
            # Calculate possible doses - handle floating point steps
            if isinstance(step, float) or isinstance(min_dose, float) or isinstance(max_dose, float):
                possible_doses = []
                current = min_dose
                while current <= max_dose:
                    possible_doses.append(current)
                    current += step
            else:
                possible_doses = list(range(min_dose, max_dose + 1, step))
                
            primary_dose = random.choice(possible_doses)
            
            # For FEP patients, often use lower doses initially
            if diagnosis == "First Episode Psychosis" and random.random() < 0.7:
                primary_dose = min(primary_dose, (min_dose + max_dose) / 2)
                primary_dose = possible_doses[min(len(possible_doses)-1, possible_doses.index(min(possible_doses, key=lambda x: abs(x-primary_dose))))]
            
            unit = MEDICATION_UNITS.get(primary_med, 'mg')
            medications.append({
                'name': primary_med,
                'category': primary_category,
                'dosage': primary_dose,
                'units': unit
            })
            
            # Small chance of adjunctive medication
            if random.random() < 0.2:
                secondary_category = random.choice(['Atypical_Secondary', 'Other'])
                secondary_med = random.choice(MEDICATION_CATEGORIES[secondary_category])
                
                min_dose, max_dose, step = MEDICATION_DOSAGES[secondary_med]
                if isinstance(step, float) or isinstance(min_dose, float) or isinstance(max_dose, float):
                    possible_doses = []
                    current = min_dose
                    while current <= max_dose:
                        possible_doses.append(current)
                        current += step
                else:
                    possible_doses = list(range(min_dose, max_dose + 1, step))
                    
                secondary_dose = random.choice(possible_doses)
                # Usually low dose for adjunctive
                secondary_dose = possible_doses[0] if len(possible_doses) > 0 else min_dose
                
                unit = MEDICATION_UNITS.get(secondary_med, 'mg')
                medications.append({
                    'name': secondary_med,
                    'category': secondary_category,
                    'dosage': secondary_dose,
                    'units': unit
                })
                
        elif protocol == 'Clozapine Pathway':
            # Clozapine as primary medication
            primary_med = 'Clozapine'
            min_dose, max_dose, step = MEDICATION_DOSAGES[primary_med]
            
            possible_doses = []
            current = min_dose
            while current <= max_dose:
                possible_doses.append(current)
                current += step
                
            # Dose depends on response - higher for non-responders
            if will_respond:
                # Lower-mid range often sufficient for responders
                primary_dose = random.choice(possible_doses[:int(len(possible_doses)*0.6)])
            else:
                # Higher doses tried for non-responders
                primary_dose = random.choice(possible_doses[int(len(possible_doses)*0.4):])
                
            medications.append({
                'name': primary_med,
                'category': 'Atypical_Secondary',
                'dosage': primary_dose,
                'units': 'mg'
            })
            
            # Chance of augmentation strategy for clozapine
            if not will_respond and random.random() < 0.4:
                aug_options = ['Aripiprazole', 'Lamotrigine', 'Valproate']
                aug_med = random.choice(aug_options)
                
                min_dose, max_dose, step = MEDICATION_DOSAGES[aug_med]
                if isinstance(step, float) or isinstance(min_dose, float) or isinstance(max_dose, float):
                    possible_doses = []
                    current = min_dose
                    while current <= max_dose:
                        possible_doses.append(current)
                        current += step
                else:
                    possible_doses = list(range(min_dose, max_dose + 1, step))
                    
                aug_dose = random.choice(possible_doses[:int(len(possible_doses)*0.6)])  # Usually lower dose
                
                aug_category = 'Atypical_Primary' if aug_med == 'Aripiprazole' else 'Mood_Stabilizers'
                unit = MEDICATION_UNITS.get(aug_med, 'mg')
                
                medications.append({
                    'name': aug_med,
                    'category': aug_category,
                    'dosage': aug_dose,
                    'units': unit
                })
                
        elif protocol == 'Long-acting Injectable':
            # Choose a LAI
            primary_category = 'LAI'
            primary_med = random.choice(MEDICATION_CATEGORIES[primary_category])
            min_dose, max_dose, step = MEDICATION_DOSAGES[primary_med]
            
            if isinstance(step, float) or isinstance(min_dose, float) or isinstance(max_dose, float):
                possible_doses = []
                current = min_dose
                while current <= max_dose:
                    possible_doses.append(current)
                    current += step
            else:
                possible_doses = list(range(min_dose, max_dose + 1, step))
                
            primary_dose = random.choice(possible_doses)
            unit = MEDICATION_UNITS.get(primary_med, 'mg')
            
            medications.append({
                'name': primary_med,
                'category': primary_category,
                'dosage': primary_dose,
                'units': unit
            })
            
            # Sometimes oral supplementation or coverage
            if random.random() < 0.3:
                # Get oral version of same medication if possible
                oral_version = primary_med.split()[0]  # Remove "Maintena", "Consta", etc.
                if oral_version in MEDICATION_DOSAGES:
                    secondary_med = oral_version
                    secondary_category = 'Atypical_Primary'
                else:
                    secondary_category = 'Atypical_Primary'
                    secondary_med = random.choice(MEDICATION_CATEGORIES[secondary_category])
                
                min_dose, max_dose, step = MEDICATION_DOSAGES[secondary_med]
                if isinstance(step, float) or isinstance(min_dose, float) or isinstance(max_dose, float):
                    possible_doses = []
                    current = min_dose
                    while current <= max_dose:
                        possible_doses.append(current)
                        current += step
                else:
                    possible_doses = list(range(min_dose, max_dose + 1, step))
                    
                secondary_dose = possible_doses[0] if len(possible_doses) > 0 else min_dose  # Low dose
                
                medications.append({
                    'name': secondary_med,
                    'category': secondary_category,
                    'dosage': secondary_dose,
                    'units': 'mg'
                })
                
        elif protocol == 'Combined AP+MS':
            # Antipsychotic + Mood Stabilizer combo (for schizoaffective or bipolar w/ psychosis)
            
            # Antipsychotic component
            primary_category = 'Atypical_Primary'
            primary_med = random.choice(MEDICATION_CATEGORIES[primary_category])
            min_dose, max_dose, step = MEDICATION_DOSAGES[primary_med]
            
            if isinstance(step, float) or isinstance(min_dose, float) or isinstance(max_dose, float):
                possible_doses = []
                current = min_dose
                while current <= max_dose:
                    possible_doses.append(current)
                    current += step
            else:
                possible_doses = list(range(min_dose, max_dose + 1, step))
                
            primary_dose = random.choice(possible_doses)
            
            medications.append({
                'name': primary_med,
                'category': primary_category,
                'dosage': primary_dose,
                'units': 'mg'
            })
            
            # Mood stabilizer component
            secondary_category = 'Mood_Stabilizers'
            secondary_med = random.choice(MEDICATION_CATEGORIES[secondary_category])
            
            min_dose, max_dose, step = MEDICATION_DOSAGES[secondary_med]
            if isinstance(step, float) or isinstance(min_dose, float) or isinstance(max_dose, float):
                possible_doses = []
                current = min_dose
                while current <= max_dose:
                    possible_doses.append(current)
                    current += step
            else:
                possible_doses = list(range(min_dose, max_dose + 1, step))
                
            secondary_dose = random.choice(possible_doses)
            unit = MEDICATION_UNITS.get(secondary_med, 'mg')
            
            medications.append({
                'name': secondary_med,
                'category': secondary_category,
                'dosage': secondary_dose,
                'units': unit
            })
    
    # Convert medications list to string format for storage in the dataframe
    if medications:
        meds_formatted = []
        for med in medications:
            meds_formatted.append(f"{med['name']} {med['dosage']}{med['units']}")
        patient['medications'] = "; ".join(meds_formatted)
        patient['medication_count'] = len(medications)
    else:
        patient['medications'] = "Aucun"
        patient['medication_count'] = 0
    
    return patient

# --- Main Data Generation Function ---
def generate_patient_data():
    """Generate main patient data"""
    logging.info("Generating patient main data...")
    patients = []
    base_start_date = START_DATE
    liste_comorbidites = [
        "HTA", "DLP", "DBTII", "Diabète", "Trouble Anxieux NS",
        "TPL", "TU ROH", "TOC", "Asthme", "Hypothyroïdie",
        "Migraine", "Syndrome de l'Intestin Irritable", 
        "Cannabis Use Disorder", "Stimulant Use Disorder"
    ]
    
    # Define risk factors list
    risk_factors_list = [
        "Antécédents Familiaux", "Traumatisme Infantile", "Complications Obstétricales",
        "Usage Précoce de Cannabis", "Urbanicité", "Migration", "Isolement Social",
        "Trauma Psychologique", "Infections Cérébrales", "Malnutrition Périnatale"
    ]

    for i in range(1, NUM_PATIENTS + 1):
        patient_id = f'P{str(i).zfill(3)}'
        # Younger age distribution for FEP patients
        age = max(16, min(45, int(np.random.normal(26.5, 7.2))))
        sex = random.choices(['1', '2'], weights=[0.60, 0.40])[0]  # Higher proportion of males in FEP
        protocol = random.choice(PROTOCOLS)
        
        # Generate Duration of Untreated Psychosis (DUP) - typically 1-36 months with most common 6-12
        if patient_id == 'P001':
            dup_months = 18  # Longer DUP for demo patient
        elif patient_id == 'P002':
            dup_months = 3   # Short DUP for demo patient
        elif patient_id == 'P003':
            dup_months = 24  # Very long DUP for demo patient
        elif patient_id == 'P004':
            dup_months = 9   # Moderate DUP for demo patient
        elif patient_id == 'P005':
            dup_months = 6   # Typical DUP for demo patient
        else:
            # Log-normal distribution with most values between 1-18 months, skewed right
            dup_months = max(1, min(36, int(np.random.lognormal(1.8, 0.8))))
        
        # Generate Risk Factors - typically 0-4 factors
        if patient_id == 'P001':
            patient_risk_factors = ["Usage Précoce de Cannabis", "Isolement Social"]
        elif patient_id == 'P002':
            patient_risk_factors = ["Antécédents Familiaux", "Traumatisme Infantile", "Trauma Psychologique"]
        elif patient_id == 'P003':
            patient_risk_factors = ["Antécédents Familiaux"]
        elif patient_id == 'P004':
            patient_risk_factors = ["Usage Précoce de Cannabis", "Migration", "Trauma Psychologique"]
        elif patient_id == 'P005':
            patient_risk_factors = ["Urbanicité", "Isolement Social"]
        else:
            # Random number of risk factors
            num_risk_factors = random.choices([0, 1, 2, 3, 4], weights=[0.15, 0.30, 0.30, 0.15, 0.10])[0]
            if num_risk_factors == 0:
                patient_risk_factors = ["Aucun"]
            else:
                patient_risk_factors = random.sample(risk_factors_list, min(num_risk_factors, len(risk_factors_list)))
        
        # Format risk factors as string
        risk_factors_str = "; ".join(patient_risk_factors)

        # Assign diagnosis based on weights
        diagnosis = random.choices(DIAGNOSES, weights=DIAGNOSIS_WEIGHTS)[0]
        
        # --- Response Logic ---
        # Determine base probabilities based on protocol
        base_response_prob = PROTOCOL_RESPONSE_RATES[protocol]
        base_functional_prob = PROTOCOL_FUNCTIONAL_RESPONSE_RATES[protocol]
        
        # Adjust probability based on diagnosis
        if diagnosis == "First Episode Psychosis":
            response_adjustment = 0.05  # Better chance for true FEP
        elif diagnosis == "Schizophrenia":
            response_adjustment = -0.1  # Worse chance for established schizophrenia
        else:
            response_adjustment = 0
        
        # Final adjusted probabilities
        adjusted_response_prob = min(0.95, max(0.05, base_response_prob + response_adjustment))
        adjusted_functional_prob = min(0.95, max(0.05, base_functional_prob + response_adjustment))
        
        # Determine clinical response (30% reduction in PANSS positive symptoms)
        will_respond = random.random() < adjusted_response_prob
        
        # Determine functional response (CGI score <= 2)
        will_have_functional_response = random.random() < adjusted_functional_prob
        
        # --- FEP-specific data generation ---
        # PANSS scores (Positive and Negative Syndrome Scale)
        # Baseline PANSS values
        panss_pos_bl = max(7, min(49, int(np.random.normal(22.5, 6.2))))  # Range 7-49
        panss_neg_bl = max(7, min(49, int(np.random.normal(19.8, 7.5))))  # Range 7-49
        panss_gen_bl = max(16, min(112, int(np.random.normal(38.5, 10.2))))  # Range 16-112
        
        # Calculate follow-up PANSS based on response status
        # Clinical response defined as 30% reduction in positive symptoms
        pos_improvement = random.uniform(0.35, 0.70) if will_respond else random.uniform(0.05, 0.25)
        panss_pos_fu = max(7, int(panss_pos_bl * (1 - pos_improvement)))
        
        # Calculate percent reduction in positive symptoms
        pos_reduction_pct = (panss_pos_bl - panss_pos_fu) / panss_pos_bl * 100
        
        # Other symptoms improve at different rates
        neg_improvement = pos_improvement * random.uniform(0.6, 0.9)  # Negative symptoms improve less
        panss_neg_fu = max(7, int(panss_neg_bl * (1 - neg_improvement)))
        
        gen_improvement = pos_improvement * random.uniform(0.7, 1.0)
        panss_gen_fu = max(16, int(panss_gen_bl * (1 - gen_improvement)))
        
        # Calculate totals
        panss_total_bl = panss_pos_bl + panss_neg_bl + panss_gen_bl
        panss_total_fu = panss_pos_fu + panss_neg_fu + panss_gen_fu
        
        # Calculate individual PANSS item scores
        pos_items_bl, neg_items_bl, gen_items_bl = distribute_panss_scores(panss_pos_bl, panss_neg_bl, panss_gen_bl)
        pos_items_fu, neg_items_fu, gen_items_fu = distribute_panss_scores(panss_pos_fu, panss_neg_fu, panss_gen_fu)
        
        # CGI scores (Clinical Global Impression)
        # CGI-S baseline (severity): 3=mild, 4=moderate, 5=marked, 6=severe
        cgi_s_bl = random.choices([3, 4, 5, 6], weights=[0.1, 0.3, 0.4, 0.2])[0]
        
        # CGI-I (improvement): 1=very much improved, 2=much improved, 3=minimally improved, 4=no change
        if will_have_functional_response:
            cgi_i = random.choices([1, 2], weights=[0.4, 0.6])[0]  # CGI 1-2 = functional response
        else:
            cgi_i = random.choices([3, 4, 5], weights=[0.5, 0.4, 0.1])[0]
        
        # Calculate CGI-S at follow-up based on CGI-I
        cgi_improvement_map = {1: -3, 2: -2, 3: -1, 4: 0, 5: 1}  # How many points of improvement for each CGI-I score
        cgi_s_fu = max(1, cgi_s_bl + cgi_improvement_map[cgi_i])
        
        # Calgary Depression Scale (0-27 range)
        calgary_bl = max(0, min(27, int(np.random.normal(8.5, 5.2))))
        calgary_improvement = pos_improvement * random.uniform(0.8, 1.2)  # Correlates somewhat with positive symptom improvement
        calgary_fu = max(0, int(calgary_bl * (1 - calgary_improvement)))
        calgary_items_bl = distribute_calgary_score(calgary_bl)
        calgary_items_fu = distribute_calgary_score(calgary_fu)
        
        # PSYRATS scores - for hallucinations and delusions
        has_hallucinations = random.random() < (0.75 if diagnosis in ["Schizophrenia", "First Episode Psychosis"] else 0.5)
        has_delusions = random.random() < (0.85 if diagnosis in ["Schizophrenia", "First Episode Psychosis", "Psychosis NOS"] else 0.6)
        
        if has_hallucinations:
            psyrats_ah_bl = max(0, min(40, int(np.random.normal(22.5, 8.5))))
            psyrats_ah_fu = max(0, int(psyrats_ah_bl * (1 - pos_improvement * random.uniform(0.8, 1.2))))
            ah_items_bl, _ = distribute_psyrats_scores(psyrats_ah_bl, 0)
            ah_items_fu, _ = distribute_psyrats_scores(psyrats_ah_fu, 0)
        else:
            psyrats_ah_bl = 0
            psyrats_ah_fu = 0
            ah_items_bl = [0] * 10
            ah_items_fu = [0] * 10
            
        if has_delusions:
            psyrats_del_bl = max(0, min(24, int(np.random.normal(16.2, 5.5))))
            psyrats_del_fu = max(0, int(psyrats_del_bl * (1 - pos_improvement * random.uniform(0.9, 1.1))))
            _, del_items_bl = distribute_psyrats_scores(0, psyrats_del_bl)
            _, del_items_fu = distribute_psyrats_scores(0, psyrats_del_fu)
        else:
            psyrats_del_bl = 0
            psyrats_del_fu = 0
            del_items_bl = [0] * 6
            del_items_fu = [0] * 6
        
        # Functional data
        hospitalizations_past_year = random.choices([0, 1, 2, 3, 4, 5], 
                                                   weights=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05])[0]
        er_visits_past_year = random.choices([0, 1, 2, 3, 4, 5, 6], 
                                             weights=[0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.05])[0]
        
        # Generate substance use data - common in FEP but with more realistic quantification
        # Cannabis use - grams per day (if used)
        cannabis_use_prob = 0.40  # 40% of FEP patients use cannabis
        cannabis_user = random.random() < cannabis_use_prob
        if cannabis_user:
            # For users: normally distributed between 0.5-7g per day
            cannabis_amount = max(0.5, min(7.0, np.random.normal(2.5, 1.5)))
            cannabis_use = round(cannabis_amount, 1)  # Round to 1 decimal place
        else:
            cannabis_use = 0
            
        # Stimulant use - pills per day (if used)
        stimulant_use_prob = 0.25  # 25% of FEP patients use stimulants
        stimulant_user = random.random() < stimulant_use_prob
        if stimulant_user:
            # For users: normally distributed between 1-5 pills per day
            stimulant_amount = max(1, min(5, np.random.normal(2.0, 1.0)))
            stimulant_use = round(stimulant_amount)  # Round to whole pills
        else:
            stimulant_use = 0
            
        # Alcohol use - standard drinks (if used)
        alcohol_use_prob = 0.50  # 50% of FEP patients use alcohol
        alcohol_user = random.random() < alcohol_use_prob
        if alcohol_user:
            # For users: normally distributed between 1-12 standard drinks
            alcohol_amount = max(1, min(12, np.random.normal(4.5, 2.5)))
            alcohol_use = round(alcohol_amount)  # Round to whole drinks
        else:
            alcohol_use = 0
        
        # Global Assessment of Functioning (GAF) - 0-100 scale
        gaf_bl = max(1, min(100, int(np.random.normal(45, 12))))
        gaf_improvement = pos_improvement * random.uniform(0.9, 1.3)
        gaf_fu = min(100, int(gaf_bl + (gaf_improvement * 30)))  # Improve by up to 30 points
        
        # Patient start date
        patient_start_date = base_start_date + timedelta(days=random.randint(0, 10) + (i // 5))

        # Create the patient dictionary with FEP-specific data
        patient = {
            'ID': patient_id,
            'age': age,
            'sexe': sex,
            'protocol': protocol,
            'diagnosis': diagnosis,
            'Timestamp': patient_start_date.strftime('%Y-%m-%d %H:%M:%S'),
            
            # PANSS scores
            'panss_pos_bl': panss_pos_bl,
            'panss_neg_bl': panss_neg_bl,
            'panss_gen_bl': panss_gen_bl,
            'panss_total_bl': panss_total_bl,
            'panss_pos_fu': panss_pos_fu,
            'panss_neg_fu': panss_neg_fu,
            'panss_gen_fu': panss_gen_fu,
            'panss_total_fu': panss_total_fu,
            'panss_pos_reduction_pct': pos_reduction_pct,
            
            # CGI scores
            'cgi_s_bl': cgi_s_bl, 
            'cgi_s_fu': cgi_s_fu,
            'cgi_i': cgi_i,
            
            # Calgary Depression Scale
            'calgary_bl': calgary_bl,
            'calgary_fu': calgary_fu,
            
            # PSYRATS scores
            'psyrats_ah_bl': psyrats_ah_bl,
            'psyrats_ah_fu': psyrats_ah_fu,
            'psyrats_del_bl': psyrats_del_bl,
            'psyrats_del_fu': psyrats_del_fu,
            
            # Functional data
            'hospitalizations_past_year': hospitalizations_past_year,
            'er_visits_past_year': er_visits_past_year,
            'gaf_bl': gaf_bl,
            'gaf_fu': gaf_fu,
            
            # Drug use
            'cannabis_use': cannabis_use,
            'stimulant_use': stimulant_use,
            'alcohol_use': alcohol_use,
            
            # Response flags
            'clinical_response': 1 if pos_reduction_pct >= 30 else 0,  # PANSS positive reduction ≥30%
            'functional_response': 1 if cgi_s_fu <= 2 else 0,  # CGI-S score of 1-2 (normal or borderline)
            
            # DUP and Risk Factors
            'dup_months': dup_months,
            'risk_factors': risk_factors_str
        }
        
        # Add detailed PANSS item scores
        for i, score in enumerate(pos_items_bl, 1):
            patient[f'panss_P{i}_bl'] = score
        for i, score in enumerate(pos_items_fu, 1):
            patient[f'panss_P{i}_fu'] = score
            
        for i, score in enumerate(neg_items_bl, 1):
            patient[f'panss_N{i}_bl'] = score
        for i, score in enumerate(neg_items_fu, 1):
            patient[f'panss_N{i}_fu'] = score
            
        for i, score in enumerate(gen_items_bl, 1):
            patient[f'panss_G{i}_bl'] = score
        for i, score in enumerate(gen_items_fu, 1):
            patient[f'panss_G{i}_fu'] = score
            
        # Add Calgary Depression Scale item scores
        for i, score in enumerate(calgary_items_bl, 1):
            patient[f'calgary_{i}_bl'] = score
        for i, score in enumerate(calgary_items_fu, 1):
            patient[f'calgary_{i}_fu'] = score
            
        # Add PSYRATS item scores
        for i, score in enumerate(ah_items_bl, 1):
            patient[f'psyrats_ah_{i}_bl'] = score
        for i, score in enumerate(ah_items_fu, 1):
            patient[f'psyrats_ah_{i}_fu'] = score
            
        for i, score in enumerate(del_items_bl, 1):
            patient[f'psyrats_del_{i}_bl'] = score
        for i, score in enumerate(del_items_fu, 1):
            patient[f'psyrats_del_{i}_fu'] = score

        # Keep some demographics from original template
        patient['pregnant'] = '0' if sex == '1' else random.choices(['0', '1'], weights=[0.98, 0.02])[0]  # Lower pregnancy rate in FEP
        patient['cigarette_bl'] = random.choices(['0', '1'], weights=[0.40, 0.60])[0]  # Higher smoking in psychosis
        patient['hospitalisation_bl'] = '1' if hospitalizations_past_year > 0 else '0'
        patient['annees_education_bl'] = max(8, min(20, int(np.random.normal(12.8, 2.9))))  # Typically lower education in FEP
        patient['revenu_bl'] = max(8000, min(120000, int(np.random.lognormal(10.2, 0.9))))  # Typically lower income in FEP

        # Assign Comorbidities
        if patient_id == 'P001': 
            patient_comorbidities_list = ["Cannabis Use Disorder"]
        elif patient_id == 'P002': 
            patient_comorbidities_list = ["Trouble Anxieux NS", "Diabète"]
        elif patient_id == 'P003': 
            patient_comorbidities_list = ["DLP", "TU ROH"]
        elif patient_id == 'P004': 
            patient_comorbidities_list = ["Cannabis Use Disorder", "TPL"]
        elif patient_id == 'P005': 
            patient_comorbidities_list = ["TU ROH", "Stimulant Use Disorder"]
        else: # Random assignment for others
            num_comorbidities = random.choices([0, 1, 2], weights=[0.30, 0.45, 0.25])[0]
            if num_comorbidities == 0: 
                patient_comorbidities_list = ["Aucune"]
            else:
                if num_comorbidities <= len(liste_comorbidites): 
                    patient_comorbidities_list = random.sample(liste_comorbidites, num_comorbidities)
                else: 
                    patient_comorbidities_list = random.choices(liste_comorbidites, k=num_comorbidities)
        
        patient['comorbidities'] = "; ".join(patient_comorbidities_list)

        # Generate medication data - pass clinical response instead of BFI-based will_respond
        patient = generate_medication_data(patient, patient['clinical_response'] == 1)
        
        patients.append(patient)

    logging.info(f"Generated main data for {len(patients)} patients.")
    return pd.DataFrame(patients)

# --- EMA, Side Effects, Nurse Notes Generation (Unchanged from previous full version) ---

def generate_ema_data(patient_df):
    # (Code as provided in previous response)
    logging.info("Generating EMA data...")
    MADRS_ITEMS = [f'madrs_{i}' for i in range(1, 11)]
    ANXIETY_ITEMS = [f'anxiety_{i}' for i in range(1, 6)]
    SLEEP, ENERGY, STRESS = 'sleep', 'energy', 'stress'
    SYMPTOMS = MADRS_ITEMS + ANXIETY_ITEMS + [SLEEP, ENERGY, STRESS]
    ema_entries = []
    if patient_df.empty: logging.warning("Patient DF empty for EMA generation."); return pd.DataFrame()
    for _, patient in patient_df.iterrows():
        patient_id = patient['ID']; protocol = patient['protocol']; improved = patient.get('will_respond', False)
        baseline_severity = patient.get('madrs_score_bl', 25) / 40.0; current_severity = baseline_severity
        protocol_effect={'HF - 10Hz': 0.8, 'BR - 18Hz': 0.65, 'iTBS': 0.5}; stability = protocol_effect.get(protocol, 0.6)
        patient_start_date = pd.to_datetime(patient['Timestamp'])
        for day in range(1, SIMULATION_DURATION_DAYS + 1):
            if random.random() < EMA_MISSING_DAY_PROB: continue
            day_effect = day / float(SIMULATION_DURATION_DAYS); improvement_factor = 0.7 if improved else 0.2
            target_severity = baseline_severity * (1 - day_effect * improvement_factor)
            noise_level = 0.15 * (1 - day_effect * 0.5)
            current_severity = current_severity * stability + target_severity * (1 - stability) + random.uniform(-noise_level, noise_level)
            current_severity = max(0, min(1, current_severity))
            n_entries_planned = random.choices([1, 2, 3], weights=EMA_ENTRIES_PER_DAY_WEIGHTS)[0]
            for entry_num in range(1, n_entries_planned + 1):
                if n_entries_planned > 1 and random.random() < EMA_MISSING_ENTRY_PROB: continue
                hour = random.randint(8, 21); minute = random.randint(0, 59)
                timestamp = patient_start_date + timedelta(days=day - 1, hours=hour, minutes=minute)
                entry_severity = max(0, min(1, current_severity * random.uniform(0.9, 1.1)))
                ema_entry = {'PatientID': patient_id, 'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'Day': day, 'Entry': entry_num}
                for item in range(1, 11): ema_entry[f'madrs_{item}'] = int(max(0, min(6, (entry_severity*6) + random.uniform(-1.5, 1.5))))
                for item in range(1, 6): ema_entry[f'anxiety_{item}'] = int(max(0, min(4, (entry_severity*4) + random.uniform(-1, 1))))
                ema_entry[SLEEP] = int(max(0, min(4, ((1-entry_severity)*4) + random.uniform(-1, 1)))); ema_entry[ENERGY] = int(max(0, min(4, ((1-entry_severity)*4) + random.uniform(-1, 1)))); ema_entry[STRESS] = int(max(0, min(4, (entry_severity*4) + random.uniform(-1, 1))))
                ema_entries.append(ema_entry)
    logging.info(f"Generated {len(ema_entries)} EMA entries.")
    return pd.DataFrame(ema_entries)

def generate_side_effects_data(patient_df):
    # (Code as provided in previous response - including specific P001-P003 profiles)
    logging.info("Generating side effects data (specific profiles for P001-P003)...")
    if not DB_INTERACTION_ENABLED: logging.warning("DB disabled, skipping SE DB insertion."); return
    num_saved = 0
    if patient_df.empty: logging.warning("Patient DF empty for SE generation."); return
    for _, patient in patient_df.iterrows():
        patient_id = patient['ID']; patient_start_date = pd.to_datetime(patient['Timestamp'])
        if patient_id == 'P001': logging.info(f"Skipping SE for P001."); continue
        elif patient_id == 'P002':
            num_reports = random.randint(1, 2); logging.info(f"Generating {num_reports} mild/mod SE report(s) for P002.")
            for i in range(num_reports):
                day_offset=random.randint(3, 15); report_date=patient_start_date+timedelta(days=day_offset)
                report_data={'patient_id':patient_id, 'report_date':report_date.strftime('%Y-%m-%d'), 'created_by':'Simulation (P002)'}
                report_data['headache']=random.choices([0,1,2],weights=[0.3,0.5,0.2])[0]; report_data['nausea']=random.choices([0,1],weights=[0.8,0.2])[0]
                report_data['scalp_discomfort']=random.choices([0,1,2],weights=[0.4,0.4,0.2])[0]; report_data['dizziness']=random.choices([0,1],weights=[0.9,0.1])[0]
                report_data['other_effects']=random.choice(['', 'Légère fatigue passagère']) if random.random()<0.3 else ''; report_data['notes']=random.choice(['Signalé.', 'Observé.'])
                try: save_side_effect_report(report_data); num_saved+=1
                except Exception as e: logging.error(f"Failed P002 SE report: {e}")
        elif patient_id == 'P003':
            num_reports = random.randint(3, 4); logging.info(f"Generating {num_reports} sig SE report(s) for P003.")
            for i in range(num_reports):
                day_offset=random.randint(2, 20); report_date=patient_start_date+timedelta(days=day_offset)
                report_data={'patient_id':patient_id, 'report_date':report_date.strftime('%Y-%m-%d'), 'created_by':'Simulation (P003)'}
                report_data['headache']=random.choices([0,1,2,3,4,5],weights=[0.1,0.2,0.3,0.2,0.1,0.1])[0]; report_data['nausea']=random.choices([0,1,2],weights=[0.5,0.3,0.2])[0]
                report_data['scalp_discomfort']=random.choices([0,1,2,3],weights=[0.2,0.3,0.3,0.2])[0]; report_data['dizziness']=random.choices([0,1,2,3],weights=[0.6,0.2,0.1,0.1])[0]
                report_data['other_effects']=random.choice(['','Fatigue marquée','Diff concentration','Acouphènes légers']) if random.random()<0.5 else ''; report_data['notes']=random.choice(['Gêne importante.', 'Pause nécessaire.', 'Estompent après 1h.'])
                try: save_side_effect_report(report_data); num_saved+=1
                except Exception as e: logging.error(f"Failed P003 SE report: {e}")
        else: # Random for others
            num_reports = random.randint(1, 5)
            for i in range(num_reports):
                day_offset=random.randint(1, SIMULATION_DURATION_DAYS); prob_cutoff=SIDE_EFFECT_PROB_INITIAL if day_offset<=SIDE_EFFECT_DECAY_DAY else SIDE_EFFECT_PROB_LATER
                if random.random()<prob_cutoff:
                    report_date=patient_start_date+timedelta(days=day_offset)
                    report_data={'patient_id':patient_id, 'report_date':report_date.strftime('%Y-%m-%d'), 'created_by':'Simulation (Random)'}
                    report_data['headache']=random.choices([0,1,2,3,4],weights=[0.6,0.2,0.1,0.05,0.05])[0]; report_data['nausea']=random.choices([0,1,2],weights=[0.8,0.15,0.05])[0]
                    report_data['scalp_discomfort']=random.choices([0,1,2,3],weights=[0.5,0.3,0.15,0.05])[0]; report_data['dizziness']=random.choices([0,1,2],weights=[0.75,0.15,0.10])[0]
                    report_data['other_effects']=random.choice(['','Fatigue légère','']) if random.random()<0.1 else ''; report_data['notes']=random.choice(['','Mentionné passé.','Tolère ok','']) if random.random()<0.2 else ''
                    try: save_side_effect_report(report_data); num_saved+=1
                    except Exception as e: logging.error(f"Failed random SE report for {patient_id}: {e}")
    logging.info(f"Generated/attempted {num_saved} SE reports.")

def generate_nurse_notes_data(patient_df):
    # (Code as provided in previous response)
    logging.info("Generating nurse notes data...")
    if not DB_INTERACTION_ENABLED: logging.warning("DB disabled, skipping nurse note DB insertion."); return
    num_saved=0
    if patient_df.empty: logging.warning("Patient DF empty for nurse notes."); return
    for _,patient in patient_df.iterrows():
        patient_id=patient['ID']; patient_start_date=pd.to_datetime(patient['Timestamp'])
        will_respond=patient.get('will_respond',False); will_remit=patient.get('will_remit',False); protocol=patient.get('protocol','Unknown')
        initial_day=random.randint(1,3); initial_note={'patient_id':patient_id,'created_by':'Simulation','goal_status':"Not Started",'objectives':f"Init {protocol}. Obj: Réduc MADRS >50%.",'tasks':"EMA. Rapporter ES.",'comments':"Motivé.",'target_symptoms':"Humeur, Anhédonie, Insomnie",'planned_interventions':f"TMS {protocol}."}
        try: save_nurse_inputs(**initial_note); num_saved+=1
        except Exception as e: logging.error(f"Failed init note {patient_id}: {e}")
        mid_day=random.randint(12,18); mid_status="In Progress"; mid_comment="Amélio légère." if random.random()<0.6 else "Stabilité."
        if not will_respond and random.random()<0.3: mid_comment="Frustration."; mid_status="On Hold"
        mid_note=initial_note.copy(); mid_note.update({'goal_status':mid_status,'comments':mid_comment,'created_by':'Simulation'})
        try: save_nurse_inputs(**mid_note); num_saved+=1
        except Exception as e: logging.error(f"Failed mid note {patient_id}: {e}")
        final_day=random.randint(28,30)
        if will_remit: final_status="Achieved"; final_comment="Rémission."
        elif will_respond: final_status="Achieved"; final_comment="Réponse >50%."
        else: final_status="Revised"; final_comment="Réponse insuffisante."
        final_note=initial_note.copy(); final_note.update({'goal_status':final_status,'comments':final_comment,'created_by':'Simulation','tasks':"Planif suivi.",'planned_interventions':"Fin protocole." if will_respond else "Réévaluation."})
        try: save_nurse_inputs(**final_note); num_saved+=1
        except Exception as e: logging.error(f"Failed final note {patient_id}: {e}")
    logging.info(f"Generated/attempted {num_saved} nurse notes.")

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting FEP Data Simulation ---")
    if DB_INTERACTION_ENABLED: 
        logging.info("Initializing DB schema...")
        initialize_database()
    else: 
        logging.warning("DB interaction disabled, schema not initialized.")

    patient_data_df = generate_patient_data()
    # No longer removing will_respond/will_remit as we've replaced them with clinical_response and functional_response
    
    patient_csv_path = os.path.join('data', 'patient_data_with_protocol_simulated.csv')
    patient_data_simple_csv_path = os.path.join('data', 'patient_data_simulated.csv')
    patient_data_df.to_csv(patient_csv_path, index=False)
    patient_data_df.to_csv(patient_data_simple_csv_path, index=False)
    logging.info(f"Saved main FEP patient data ({len(patient_data_df)}) to CSVs.")

    # Generate EMA data
    ema_data_df = generate_ema_data(patient_data_df)
    ema_csv_path = os.path.join('data', 'simulated_ema_data.csv')
    ema_data_df.to_csv(ema_csv_path, index=False)
    logging.info(f"Saved {len(ema_data_df)} EMA entries.")

    # Generate side effects and nurse notes
    generate_side_effects_data(patient_data_df)
    generate_nurse_notes_data(patient_data_df)

    # Create config file
    config_content = f"""
paths:
  patient_data_with_protocol: "{patient_csv_path}"
  patient_data: "{patient_data_simple_csv_path}"
  simulated_ema_data: "{ema_csv_path}"
mappings:
  panss_positive_items:
    P1: Délires
    P2: Désorganisation Conceptuelle
    P3: Hallucinations
    P4: Excitation
    P5: Idées de grandeur
    P6: Méfiance/Persécution
    P7: Hostilité
  panss_negative_items:
    N1: Affect Émoussé
    N2: Retrait Émotionnel
    N3: Mauvais Rapport
    N4: Retrait Social Passif
    N5: Difficultés d'Abstraction
    N6: Fluidité de Conversation
    N7: Pensée Stéréotypée
  cgi_severity:
    1: Normal
    2: À la limite
    3: Légèrement malade
    4: Modérément malade
    5: Manifestement malade
    6: Gravement malade
    7: Parmi les plus malades
  cgi_improvement:
    1: Très fortement amélioré
    2: Fortement amélioré
    3: Légèrement amélioré
    4: Pas de changement
    5: Légèrement aggravé
    6: Fortement aggravé
    7: Très fortement aggravé
"""
    os.makedirs('config', exist_ok=True)
    config_path = os.path.join('config', 'config.yaml')
    try:
        with open(config_path, 'w', encoding='utf-8') as f: 
            f.write(config_content)
        logging.info(f"Updated {config_path}")
    except Exception as e: 
        logging.error(f"Failed to write {config_path}: {e}")
        
    logging.info("--- FEP Simulation complete. ---")
    print("\nSimulation complete. Run app with: 'streamlit run app.py'")
    print(f"Log file: {log_file}")