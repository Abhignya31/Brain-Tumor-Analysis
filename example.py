import google.generativeai as genai

genai.configure(api_key="AIzaSyA5Mb_NNkxW6iUIdE4duPTpj3f3EdEwnRA")

models = genai.list_models()

for m in models:
    print(m.name, " → ", m.supported_generation_methods)
