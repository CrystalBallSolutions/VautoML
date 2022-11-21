import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format
from pycaret.regression import setup, compare_models, pull, save_model, convert_model


def make_model(df:pd.DataFrame, target:str, seed:int=42) -> tuple:
    setup(df, target=target, silent=True, session_id=seed)
    best_model = compare_models()
    compare_df = pull()
    save_model(best_model, 'best_model')

    if __name__ == '__main__':
        print('Compare trials:\n', compare_df)
        print('Best Model:\n', best_model)
        try:
            print(convert_model(best_model, 'python'), file=open('model.py','w'))
            print('Equivalent Python model written to model.py')
        except:
            pass
    else:
        import streamlit as st
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download the model file', f, 'trained_model.pkl')
        st.write('Compare trials:')
        st.dataframe(compare_df)
        st.write('Best Model:')
        st.write(best_model)
        try:
            print(convert_model(best_model, 'python'), file=open('model.py','w'))
            st.info('Equivalent Python model written to model.py (if possible)')
            with open('model.py', 'rb') as f:
                st.download_button('Download the Python file', f, 'model.py')
        except:
            pass
        
if __name__ == '__main__':
    # get test data
    from pycaret.datasets import get_data
    df = get_data('diamond')
    make_model(df, 'Price')

