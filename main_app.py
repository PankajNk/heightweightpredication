from flask import Flask ,request,render_template
import pickle 
import numpy as np

app = Flask(__name__)
loaded_model = pickle.load(open("Weight_Height_Model.pkl","rb"))

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/getpredication",methods=['POST'] )
def getpredication():
	input = [float(x) for x in request.form.values()]
	final_input = [np.array(input)]
	print(f"final_input",final_input)
	predication = loaded_model.predict(final_input)
	print(predication)
	output =np.round(predication, 2)
	print(output)
	

	return render_template("index.html",prediction_text = "Predicted weight is {}".format(output))




if __name__ == "__main__":
	app.run(debug=True)
