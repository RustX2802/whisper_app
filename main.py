from app import *

if __name__ == '__main__':
    config()

    if st.session_state['page_index'] == -1:
        # Specify token page (mandatory to use the diarization option)
        st.warning('You must specify a token to use the diarization model. Otherwise, the app will be launched without this model. You can learn how to create your token here: https://huggingface.co/pyannote/speaker-diarization / 분할 모델을 사용하려면 토큰을 지정해야 합니다. 그렇지 않으면 이 모델 없이 앱이 실행됩니다. 여기에서 토큰을 만드는 방법을 배울 수 있습니다: https://huggingface.co/pyannote/speaker-diarization')
        text_input = st.text_input("Enter your Hugging Face token: / Hugging Face 토큰을 입력하세요:", placeholder="ACCESS_TOKEN_GOES_HERE", type="password")

        # Confirm or continue without the option
        col1, col2 = st.columns(2)

        # Save changes button
        with col1:
            confirm_btn = st.button("I have changed my token / 토큰을 변경했습니다", on_click=confirm_token_change, args=(text_input, 0), disabled=st.session_state["disable"])
            # if text is changed, button is clickable
            if text_input != "ACCESS_TOKEN_GOES_HERE":
                st.session_state["disable"] = False

        # Continue without a token (there will be no diarization option)
        with col2:
            dont_mind_btn = st.button("Continue without this option / 이 옵션 없이 계속하십시오", on_click=update_session_state, args=("page_index", 0))

    if st.session_state['page_index'] == 0:
        # Home page
        choice = st.radio("Features / 특징", ["By a video URL / 비디오 URL로", "By uploading a file / 파일을 업로드하여"]) 

        stt_tokenizer, stt_model, summarizer, dia_pipeline = load_models()

        if choice == "By a video URL / 비디오 URL로":
            transcript_from_url(stt_tokenizer, stt_model, summarizer, dia_pipeline)

        elif choice == "By uploading a file / 파일을 업로드하여":
            transcript_from_file(stt_tokenizer, stt_model, summarizer, dia_pipeline)

    elif st.session_state['page_index'] == 1:
        # Display Results page
        display_results()

    elif st.session_state['page_index'] == 2:
        # Rename speakers page
        rename_speakers_window()