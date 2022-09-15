from VideoMaker import VideoMaker

editor = VideoMaker("assets/myphoto.png", "assets/thanos.mp4", (960, 720))
editor.resizeVideo((512, 288))
editor.makeVideo1(350, 370)