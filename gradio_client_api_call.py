from gradio_client import Client, handle_file

# client = Client("CraigRichards/fastai_pet_clasifier")
client = Client("ShalabhKamboj/fastai_pet_detector")
result = client.predict(
		img=handle_file('snowflake.jpg'),
		api_name="/predict"
)
print(result)