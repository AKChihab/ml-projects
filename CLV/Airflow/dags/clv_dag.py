from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from clv_predictive_mlops_project import generate_mock_data, build_features, compute_targets, train_clv_model, train_upsell_model

default_args = {
    'start_date': datetime(2023,1,1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('clv_pipeline',
         schedule_interval='@monthly',
         default_args=default_args,
         catchup=False) as dag:

    task_gen = PythonOperator(
        task_id='generate_data',
        python_callable=generate_mock_data
    )
    task_feat = PythonOperator(
        task_id='build_features',
        python_callable=build_features,
        op_kwargs={'customers': '{{ ti.xcom_pull("generate_data")[0] }}',
                   'transactions': '{{ ti.xcom_pull("generate_data")[1] }}',
                   'observation_end': datetime(2022,12,31)}
    )
    task_target = PythonOperator(
        task_id='compute_targets',
        python_callable=compute_targets,
        op_kwargs={'transactions': '{{ ti.xcom_pull("generate_data")[1] }}',
                   'observation_end': datetime(2022,12,31)}
    )
    task_clv = PythonOperator(
        task_id='train_clv',
        python_callable=train_clv_model,
        op_kwargs={'features': '{{ ti.xcom_pull("build_features") }}',
                   'targets': '{{ ti.xcom_pull("compute_targets") }}'}
    )
    task_upsell = PythonOperator(
        task_id='train_upsell',
        python_callable=train_upsell_model,
        op_kwargs={'features': '{{ ti.xcom_pull("build_features") }}',
                   'targets': '{{ ti.xcom_pull("compute_targets") }}'}
    )

    task_gen >> [task_feat, task_target] >> [task_clv, task_upsell]
