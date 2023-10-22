A project to dockerize [clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator), a tool for captioning images.
It is intended for offline usage, pre-bundled with the clip model.

It optionally allows the user to save the description to the exif metadata of the image.

```bash
docker build -t clip-interrogator .
docker run -it --rm --network none -v $PWD/images:/home/python/images:ro clip-interrogator
```

# Images

- Cuba: https://unsplash.com/photos/people-in-traditional-dress-walking-on-street-during-daytime-MU6Z-zj1n6c
- Beach: https://unsplash.com/photos/people-standing-on-seashore-_hg9QTNFFWo
- Kopenhagen: https://unsplash.com/photos/a-row-of-boats-sitting-next-to-each-other-on-a-body-of-water-idnpWAlqVgE
