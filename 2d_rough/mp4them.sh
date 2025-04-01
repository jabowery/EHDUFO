ffmpeg -framerate 30 -i output/frames/air_flow/frame_%04d.png -framerate 30 -i output/frames/charge_density/frame_%04d.png -framerate 30 -i output/frames/streamlines/frame_%04d.png -filter_complex "[0:v]crop=660:1400:400:0[a]; [1:v]crop=660:1400:400:0[b]; [2:v]crop=660:1400:400:0[c]; [a][b][c]hstack=inputs=3" -c:v libx264 -pix_fmt yuv420p output.mp4

