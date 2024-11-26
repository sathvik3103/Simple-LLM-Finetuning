def get_medical_data():
    data = [
        {
            "input": "What are the common symptoms of type 2 diabetes?",
            "output": "Common symptoms of type 2 diabetes include increased thirst, frequent urination, blurred vision, slow-healing wounds, unexplained weight loss, fatigue, and numbness or tingling in hands or feet. It's important to note that some people may not experience any symptoms in the early stages of the disease[1]."
        },
        {
            "input": "How is hypertension diagnosed?",
            "output": "Hypertension (high blood pressure) is diagnosed through blood pressure measurements. A healthcare provider typically takes multiple readings over time. A consistent blood pressure of 130/80 mm Hg or higher is considered hypertension. Diagnosis may also involve reviewing medical history, physical examination, and sometimes additional tests like blood tests or electrocardiogram (ECG) to check for underlying conditions or complications[2]."
        },
        {
            "input": "What are the risk factors for developing osteoporosis?",
            "output": "Risk factors for osteoporosis include age (especially over 50), being female, family history, low body weight, smoking, excessive alcohol consumption, lack of exercise, low calcium and vitamin D intake, certain medical conditions (like rheumatoid arthritis or celiac disease), and use of certain medications (such as long-term corticosteroids). Ethnicity can also play a role, with Caucasian and Asian individuals at higher risk[1]."
        },
        {
            "input": "How is strep throat diagnosed and treated?",
            "output": "Strep throat is typically diagnosed through a combination of physical examination and a rapid strep test or throat culture. Treatment usually involves antibiotics, most commonly penicillin or amoxicillin, to kill the bacteria. Over-the-counter pain relievers can help manage symptoms. It's important to complete the full course of antibiotics to prevent complications and reduce the risk of spreading the infection[2]."
        },
        {
            "input": "What are the early signs of Alzheimer's disease?",
            "output": "Early signs of Alzheimer's disease may include memory loss that disrupts daily life, challenges in planning or solving problems, difficulty completing familiar tasks, confusion with time or place, trouble understanding visual images and spatial relationships, new problems with words in speaking or writing, misplacing things and losing the ability to retrace steps, decreased or poor judgment, withdrawal from work or social activities, and changes in mood and personality[1]."
        },
        {
            "input": "How is asthma diagnosed in children?",
            "output": "Diagnosing asthma in children involves several steps: reviewing symptoms and medical history, physical examination, lung function tests (for older children), and sometimes allergy tests. Doctors may use spirometry to measure lung function or peak flow meters to assess airway obstruction. In younger children, diagnosis can be challenging and may require observing response to asthma medications. Family history and presence of allergies or eczema can also provide clues[2]."
        },
        {
            "input": "What are the treatment options for rheumatoid arthritis?",
            "output": "Treatment options for rheumatoid arthritis include disease-modifying antirheumatic drugs (DMARDs) like methotrexate, biologic agents, nonsteroidal anti-inflammatory drugs (NSAIDs), corticosteroids, and pain relievers. Physical therapy, occupational therapy, and lifestyle changes such as exercise and stress management are also important. In severe cases, surgery may be considered. The goal is to reduce inflammation, relieve symptoms, prevent joint damage, and improve quality of life[1]."
        },
        {
            "input": "How is melanoma skin cancer detected?",
            "output": "Melanoma skin cancer is primarily detected through visual examination of the skin. The ABCDE rule is often used: Asymmetry, Border irregularity, Color variations, Diameter larger than 6mm, and Evolving size, shape, or color. Regular self-examinations and professional skin checks are recommended. If a suspicious mole is found, a biopsy is performed for definitive diagnosis. Early detection is crucial for successful treatment[2]."
        },
        {
            "input": "What are the stages of chronic kidney disease?",
            "output": "Chronic kidney disease (CKD) is classified into five stages based on the glomerular filtration rate (GFR) and the presence of kidney damage. Stage 1: Normal or high GFR (≥90 mL/min/1.73 m²) with evidence of kidney damage. Stage 2: Mildly reduced GFR (60-89 mL/min/1.73 m²). Stage 3: Moderately reduced GFR (30-59 mL/min/1.73 m²). Stage 4: Severely reduced GFR (15-29 mL/min/1.73 m²). Stage 5: Kidney failure (GFR <15 mL/min/1.73 m² or dialysis)[1]."
        },
        {
            "input": "How is celiac disease diagnosed?",
            "output": "Celiac disease diagnosis typically involves several steps: blood tests to check for certain antibodies, genetic testing for HLA-DQ2 and HLA-DQ8 genes, and an intestinal biopsy. The blood tests look for anti-tissue transglutaminase antibodies and other celiac-specific antibodies. If blood tests and genetic tests suggest celiac disease, an endoscopy with small intestine biopsy is usually performed to confirm the diagnosis. It's important to continue eating gluten before and during the diagnostic process for accurate results[2]."
        }
    ]
    return data

import lamini
from lamini import Lamini
import os
from dotenv import load_dotenv
load_dotenv()
lamini_api_key=os.getenv("LAMINI_API_KEY")

lamini.api_key = lamini_api_key

llm = Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

data = get_medical_data()

llm.tune(data_or_dataset_id=data,
         finetune_args={'learning_rate': 5e-5,
                        'max_steps': 1000,
                        'early_stopping': True,
                        'optim': 'sgd'})