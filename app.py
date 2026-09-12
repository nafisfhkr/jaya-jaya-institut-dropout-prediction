from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / 'model' / 'dropout_pipeline.joblib'
REQUIRED_BUNDLE_KEYS = {
    'pipeline', 'feature_columns', 'feature_defaults', 'categorical_columns',
    'category_values', 'model_name', 'decision_threshold', 'positive_class',
}


@st.cache_resource
def load_model_bundle(path: str) -> dict[str, Any]:
    bundle = joblib.load(path)
    validate_model_bundle(bundle)
    return bundle


def validate_model_bundle(bundle: dict[str, Any]) -> None:
    missing_keys = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing_keys:
        raise ValueError(f"Model bundle tidak valid; key hilang: {sorted(missing_keys)}")

    for column in bundle['categorical_columns']:
        allowed_values = bundle['category_values'].get(column, [])
        if not allowed_values:
            raise ValueError(f"Model bundle tidak memiliki kategori untuk: {column}")
        if bundle['feature_defaults'][column] not in allowed_values:
            raise ValueError(f"Default kategori tidak valid untuk: {column}")


def build_input_frame(bundle: dict[str, Any], values: dict[str, int | float]) -> pd.DataFrame:
    missing_columns = set(bundle['feature_columns']) - set(values)
    if missing_columns:
        raise ValueError(f"Input belum lengkap: {sorted(missing_columns)}")
    for column in bundle['categorical_columns']:
        if values[column] not in bundle['category_values'][column]:
            raise ValueError(f"Kategori tidak valid untuk {column}: {values[column]}")
    return pd.DataFrame([values]).reindex(columns=bundle['feature_columns'])


def group_feature_columns(feature_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    demographics, first_semester, second_semester_and_economic = [], [], []
    for column in feature_columns:
        if '1st_sem' in column:
            first_semester.append(column)
        elif '2nd_sem' in column or column in {'Unemployment_rate', 'Inflation_rate', 'GDP'}:
            second_semester_and_economic.append(column)
        else:
            demographics.append(column)
    return demographics, first_semester, second_semester_and_economic


def render_feature_inputs(
    columns: list[str], bundle: dict[str, Any], input_values: dict[str, int | float]
) -> None:
    widgets = st.columns(3)
    for index, column in enumerate(columns):
        with widgets[index % 3]:
            label = column.replace('_', ' ').title()
            default = bundle['feature_defaults'][column]
            if column in bundle['categorical_columns']:
                options = bundle['category_values'][column]
                input_values[column] = int(
                    st.selectbox(label, options, index=options.index(default), key=f'input_{column}')
                )
            else:
                input_values[column] = float(
                    st.number_input(label, value=float(default), key=f'input_{column}')
                )


def main() -> None:
    st.set_page_config(page_title='Prediksi Risiko Dropout', page_icon='🎓', layout='wide')
    st.title('🎓 Prediksi Risiko Dropout Mahasiswa - Jaya Jaya Institut')
    st.write('Aplikasi ini mengestimasi risiko dropout berdasarkan faktor demografi, akademik, dan sosial ekonomi.')
    if not MODEL_PATH.exists():
        st.error('Model belum tersedia. Jalankan notebook.ipynb terlebih dahulu.')
        st.stop()

    try:
        bundle = load_model_bundle(str(MODEL_PATH))
    except (OSError, ValueError) as error:
        st.error(f'Model tidak dapat digunakan: {error}')
        st.stop()

    demographics, first_semester, second_semester_and_economic = group_feature_columns(bundle['feature_columns'])
    st.info('Prediksi ini adalah sinyal awal untuk prioritas intervensi, bukan keputusan akademik otomatis.')
    with st.form('prediction_form'):
        input_values: dict[str, int | float] = {}
        with st.expander('📋 Data Pendaftaran dan Demografi', expanded=True):
            render_feature_inputs(demographics, bundle, input_values)
        with st.expander('📚 Data Akademik Semester 1'):
            render_feature_inputs(first_semester, bundle, input_values)
        with st.expander('📈 Data Akademik Semester 2 dan Ekonomi'):
            render_feature_inputs(second_semester_and_economic, bundle, input_values)
        submitted = st.form_submit_button('Prediksi risiko dropout', use_container_width=True)

    if submitted:
        input_frame = build_input_frame(bundle, input_values)
        dropout_probability = float(bundle['pipeline'].predict_proba(input_frame)[:, 1][0])
        threshold = float(bundle['decision_threshold'])
        risk_label = 'Risiko tinggi' if dropout_probability >= threshold else 'Risiko lebih rendah'
        st.divider()
        st.subheader('Hasil Prediksi')
        st.metric('Probabilitas dropout', f'{dropout_probability:.1%}')
        if dropout_probability >= threshold:
            st.warning(f'{risk_label}: sarankan follow-up akademik atau finansial.')
        else:
            st.success(f'{risk_label}: tetap lakukan pemantauan berkala.')
        st.caption(f"Model: {bundle['model_name']} | Decision Threshold: {threshold:.2f}")


if __name__ == '__main__':
    main()
