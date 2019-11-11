const imageUpload = document.getElementById('imageUpload')

// gọi thư viện
Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
  //
  const container = document.createElement('div')
  container.style.position = 'relative'
  document.body.append(container)
  // tảo thẻ div chứa hình


  // load bộ test 
  const labeledFaceDescriptors = await loadLabeledImages()
   //0.6 = 60% độ giống trong khuôn mặt
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  


  let image
  let canvas
  document.body.append('Loaded')
  imageUpload.addEventListener('change', async () => {
    if (image) image.remove()
    if (canvas) canvas.remove()

    // lấy hình từ file
    image = await faceapi.bufferToImage(imageUpload.files[0])
    //ad hình vào container bỏ vào body
    container.append(image)

    // add nhận diện khuôn mặt vào hình
    canvas = faceapi.createCanvasFromMedia(image)
    container.append(canvas)
    // kích cỡ
    const displaySize = { width: image.width, height: image.height }
    faceapi.matchDimensions(canvas, displaySize)

    // đếm số khuôn mặt có trong hình
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
    console.log("có "+detections.length+" khuôn mặt")
    //


    // làm ô vuông và lấy  các khuôn mặt
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    // so khớp với các hình đã train 
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
    // duyện result thay vì resizedDetections vì result đã duyệt và gắn tên các nhân vật lên rồi
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box
      //const drawBox = new faceapi.draw.DrawBox(box, { label: 'Face' })

      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      console.log(result.toString())
      drawBox.draw(canvas)
    })
  })
}

function loadLabeledImages() {
  const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes','Linh','Phèn','Thiện','Thor', 'Tony Stark']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      // duyệt hình
      for (let i = 1; i <= 2; i++) {
        // lấy hình kiểm tra
        //const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`)
        const img = await faceapi.fetchImage(`/labeled_images/${label}/${i}.jpg`)
        // lấy khuôn mặt trong list
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        // đẩy khuôn mặt vào danh sách rồi đẩy lên để kiếm
        descriptions.push(detections.descriptor)
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}
