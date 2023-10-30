A CLI tool for [clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator), a tool for captioning images.
Dockerized and intended for offline usage, pre-bundled with the clip model.

It optionally allows the user to save the description to the exif metadata of the image.

```bash
docker build -t clip-interrogator .
docker run -it --rm --network none -v $PWD/images:/home/python/images:ro clip-interrogator "./images/**/*.*"
```

# Images

- Cuba: https://unsplash.com/photos/people-in-traditional-dress-walking-on-street-during-daytime-MU6Z-zj1n6c
- Beach: https://unsplash.com/photos/people-standing-on-seashore-_hg9QTNFFWo
- Kopenhagen: https://unsplash.com/photos/a-row-of-boats-sitting-next-to-each-other-on-a-body-of-water-idnpWAlqVgE

Descriptions generated by clip_interrogator:

```
daniel-j-schwarz-idnpWAlqVgE-unsplash.jpg: a row of buildings next to a body of water, denmark, nordic pastel colors, outdoors european cityscape, by Albert Bertelsen, scandinavia, by Arvid Nyholm, peaked wooden roofs, small scandinavian!!! houses, colorful buildings, brightly colored buildings, floating buildings, european buildings, by Niels Lergaard, swedish houses, featured on unsplash, european palette
daniel-sessler-MU6Z-zj1n6c-unsplash.jpg: a group of people that are standing in the street, cuban women in havana, cuban revolution, cuban setting, cuba, people dancing in background, anthropological photography, 4k press image, travel ad, people dancing, today\'s featured photograph 4k, archival pigment print, laura letinsky and steve mccurry, photo taken with ektachrome
oscar-nord-_hg9QTNFFWo-unsplash.jpg: a crowded beach filled with lots of people, martin parr, italian beach scene, crowded beach, beautiful italian beach scene, happy italian beach scene, beach scene, beach aesthetic, in summer, beach setting, an island made of caviar, photo taken with ektachrome, children playing at the beach, moody : : wes anderson, people in beach, by Hiroshi Nagai
```
