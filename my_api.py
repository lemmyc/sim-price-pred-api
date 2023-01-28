from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import lstm_model
from flask import jsonify
# Khởi tạo Flask Server Backend
app = Flask(__name__)



# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"



model = lstm_model.lstm_model("./weights/ckpt_best.hdf5")


@app.route('/', methods=['POST'] )
def predict_sim_price():
    data = request.get_json()

    if data:
        input = data["sim_number"]

        result = {}
        result['career'] = model.predict('career', input)
        result['price_category'] = str(model.predict('price',input))
        
        return result

    return "Error while calling API"

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='7777')