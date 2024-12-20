import psycopg
from psycopg import sql
from typing import Union

def patient_search(CONNECTION: str, img_num: int) -> list:
    with psycopg.connect(CONNECTION) as conn:
        query = sql.SQL(''' 
            SELECT *
            FROM myschema.patient
            where id = {img_num}
        ''').format(img_num=sql.Literal(img_num))
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
    
    return results

def brain_search(CONNECTION: str, symptom: str) -> list:
    with psycopg.connect(CONNECTION) as conn:
        query = sql.SQL(''' 
            SELECT *
            FROM myschema.brain
            where symptom = {symptom}
        ''').format(symptom=sql.Literal(symptom))
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
    
    return results

def hospital_search(CONNECTION: str, patient_data: str) -> list:
    with psycopg.connect(CONNECTION) as conn:
        query = sql.SQL(''' 
            SELECT *
            FROM myschema.hospital
            where brain = {patient_data}
        ''').format(patient_data=sql.Literal(patient_data))
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
    
    return results