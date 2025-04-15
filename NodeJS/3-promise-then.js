const axios = require('axios')

const URL = "https://jsonplaceholder.typicode.com/posts/1"


axios
    .get(URL)
    .then((response) => {
        console.log(response.data)
    })
    .catch((error) => {
        console.error(error)
    })


console.log('This is the end of the code.')


