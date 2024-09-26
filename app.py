import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# クラスラベルの設定
classes = ["伊予松山城", "掛川城", "丸亀城","鬼ノ城","金ヶ崎城","熊本城","犬山城","五稜郭","高知城","佐賀城","鹿児島城","若松城","首里城","小田原城","松本城","上田城","大阪城","唐津城","島原城","徳島城","二条城","萩城","備中松山城","姫路城","浜松城","伏見城","福岡城","福山城","名古屋城","躑躅ヶ崎館"]
image_size = 224  # モデルが期待する224x224に変更

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='/static')

# ファイルの拡張子確認関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 学習済みモデルのロード
model = load_model('./model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print(filepath)

            # 受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, target_size=(image_size, image_size))  # サイズを224x224に変更
            img = image.img_to_array(img)
            img = img / 255.0
            data = np.expand_dims(img, axis=0)  # 画像をバッチ対応の形に変更

            # モデルに渡して予測
            result = model.predict(data)[0]
            print(result)
            predicted = result.argmax()
            print(predicted)
            pred_answer = "これは " + classes[predicted] +  " の画像です"

            return render_template("index.html", answer=pred_answer, imagefile=filename)

    return render_template("index.html", answer="")

if __name__ == "__main__":
    app.run(debug=True)
