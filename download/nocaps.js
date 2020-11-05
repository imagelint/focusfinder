const fetch = require('node-fetch')
const fs = require('fs')
console.log('start')
fetch('https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json')
    .then(res => res.json())
    .then(async function (json) {
            for (let i = 0; i < json.images.length; i++) {
                console.log(i)
                const image = json.images[i]
                if (image.height < 224 || image.width < 224) {
                    continue
                }
                const skipExisting = false
                if(skipExisting && fs.existsSync('./images/raw_images/nocaps/' + image.file_name)) {
                    continue
                }
                console.log(i)
                let request = fetch(image.coco_url)
                    .then(res => {
                        const dest = fs.createWriteStream('./images/raw_images/nocaps/' + image.file_name)
                        res.body.pipe(dest)
                    })
                if (i%40===0){
                    await request
                }
            }
        })

