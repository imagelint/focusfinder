const fetch = require('node-fetch')
const fs = require('fs')
fetch('https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json')
    .then(res => res.json())
    .then(async function (json) {
            for (let i = 0; i < json.images.length; i++) {
                console.log(i)
                const image = json.images[i]
                if (image.height < 224 || image.width < 224) {
                    continue
                }
                let request = fetch(image.coco_url)
                    .then(res => {
                        const dest = fs.createWriteStream('./images/' + image.file_name)
                        res.body.pipe(dest)
                    })
                if (i%10===0){
                    await request
                }
            }
        })

