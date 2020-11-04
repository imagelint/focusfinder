const fetch = require('node-fetch')
const fs = require('fs')
const tsv = require('tsv')
// Get the photos.tsv file from https://github.com/unsplash/datasets
// The free "Lite dataset" contains it.
const data = tsv.parse(fs.readFileSync('./photos.tsv000', 'utf8'))
let download = async function() {
    for (let i = 0; i < data.length; i++) {
        console.log(i)
        const entry = data[i]
        let request = fetch(entry.photo_image_url + '?w=244&h=244&fit=fill&fill=solid&fill-color=000000')
            .then(res => {
                const dest = fs.createWriteStream('./unsplash-images/' + entry.photo_id + '.jpg')
                res.body.pipe(dest)
            })
        if (i%40===0){
            await request
        }
    }
}

download()