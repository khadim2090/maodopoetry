import streamlit as st 
from  PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku

from tensorflow.keras.models import load_model
import tensorflow as tf

import numpy as np
model = tf.keras.models.load_model('my_model.h5')

 
image = Image.open("maodo.jpg")
st.image(image, caption='El Hadji Malick Sy'  )
 

def main():

	st.title(" Maodo Poetry   ") 
    
	max_sequence_len = 50

	tokenizer = Tokenizer()
	data = open('scrapping.txt',encoding='utf-8').read()

	corpus = data.lower().split('\n')

	tokenizer.fit_on_texts(corpus)

	text = st.text_input("Enter your text")
	seed_text = text 
	length = st.slider(label = 'Poetry length',min_value=20, max_value=100)
	next_words = length
	if st.button("Generate Poetry"):
		with st.spinner("wait.. wait... "):
			for _ in range(next_words):
				token_list = tokenizer.texts_to_sequences([seed_text])[0]
				token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
				predicted = model.predict_classes(token_list, verbose=0)			 
				output_word = ""
				for word, index in tokenizer.word_index.items():
					if index == predicted:
						output_word = word
						break
				seed_text += " " + output_word
		st.success(seed_text)



	 
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

 

if __name__ == '__main__':
	main()