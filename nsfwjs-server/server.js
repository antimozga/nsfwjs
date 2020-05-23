const express = require('express')
const multer = require('multer')
const jpeg = require('jpeg-js')

const tf = require('@tensorflow/tfjs-node')
const nsfw = require('../dist')

const app = express()
const upload = multer()

let _model

const convert = async (img) => {
  // Decoded image in UInt8 Byte array
  const image = await jpeg.decode(img, true)

  const numChannels = 3
  const numPixels = image.width * image.height
  const values = new Int32Array(numPixels * numChannels)

  for (let i = 0; i < numPixels; i++)
    for (let c = 0; c < numChannels; ++c)
      values[i * numChannels + c] = image.data[i * 4 + c]

  return tf.tensor3d(values, [image.height, image.width, numChannels], 'int32')
}

const convert2 = async (img) => {
  // Decoded image in UInt8 Byte array
//  const image = await jpeg.decode(img, true)
  try {
    return await tf.node.decodeImage(img, 3)
  } catch {
    return null;
  }
}

app.post('/nsfw', upload.single("image"), async (req, res) => {
  if (!req.file)
    res.status(400).send("Missing image multipart/form-data")
  else {
    const image = await convert2(req.file.buffer)
    if (image != null) {
	const predictions = await _model.classify(image)
	image.dispose()
	res.json(predictions)
    } else {
	res.status(400).send("Unsupported image file");
    }
  }
})

const load_model = async () => {
  _model = await nsfw.load('file:///root/nsfwjs/example/nsfw_demo/public/quant_mid/', { type: 'graph' })
}

// Keep the model in memory, make sure it's loaded only once
load_model().then(() => app.listen(8080))
