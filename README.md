# neuro ascii

this project is an experiment in bounded perception. it is a neural network that wants to be an artist, but it lives in a severely restricted world.

instead of seeing high-resolution pixels, it conceptualizes concepts by casting them into a crude 3d tri-plane environment. it takes human ideas, translates them into spatial forms, and then collapses them back into rigid, grayscale ascii characters. 

the model doesn't just draw blindly. it is equipped with a small text-decoder module that forces it to rationalize what it is doing. it thinks out loud, doubts its own work, and argues with itself. if it realizes the generated structure doesn't match the original prompt, it actively recalculates its projection planes until it feels the ascii representation is acceptable.

### training and evolution

the initial versions of this model were essentially blind. they were trained on simple image-to-ascii mappings, treating symbols merely as pixel brightness. 

for v2, the architecture was fundamentally shifted. the model was taught depth. it extracts features from images and projects them onto three intersecting planes, creating a pseudo-3d understanding of the object. the training involved teaching it to extract these spatial embeddings and decode them from multiple camera angles.

it's not a general AI. its domain is exclusively learning how light and shape translate into text grids. it has a small associative memory bank where it stores past generations, continually evaluating if its current work is novel or something it has seen before. its worldview is literally limited to the depth it can represent within its character palette.

### running it

the core pipeline is built on pytorch and exposed via a fastapi server. an instance of this model is hosted remotely.

if you want to run it yourself:
1. create a `.env` file with `UNSPLASH_ACCESS_KEY` and `PEXELS_API_KEY` to let it fetch visual references.
2. start the backend with `python app.py`.

this repository also includes an `index.html` client interface.

### hosting on github pages

to host the interface side of this project on github pages:
1. go to the settings of your repository on github.
2. navigate to the "pages" section on the left sidebar.
3. select `main` (or whichever branch you push this index to) as the source, and set the folder to `/ (root)`.
4. save. github will provide a public link where you can watch the model think and draw directly in the browser.
