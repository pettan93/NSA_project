from subprocess import check_output

RESOURCES_DIR_PATH = '../resources/'


def generate_captcha(text, output_name, settings_dict=None):
    settings_args = ["height",  # default 60
                     "width",  # default (number of text characters * 60)
                     "fontSize",  # default 60
                     "rotationAmplitude",  # default 30
                     "scaleAmplitude",  # default 40
                     "showGrid",  # default False
                     "gridSize"  # defuult 10
                     ]

    if settings_dict is None:
        settings_dict = dict()
    for arg in settings_args:
        if arg not in settings_dict:
            settings_dict[arg] = "d"  # "d" stands for "default"

    check_output(
        "java -jar "
        + RESOURCES_DIR_PATH + "captcha_generator.jar "
        + text + " "
        + RESOURCES_DIR_PATH + "/output/" + output_name + " "
        + RESOURCES_DIR_PATH + "fonts/boxpot.ttf "
        + settings_dict["height"] + " "
        + settings_dict["width"] + " "
        + settings_dict["fontSize"] + " "
        + settings_dict["rotationAmplitude"] + " "
        + settings_dict["scaleAmplitude"] + " "
        + settings_dict["showGrid"] + " "
        + settings_dict["gridSize"] + "",
        shell=True)


# usages
generate_captcha("foo", "out1.png", {"showGrid": "true", "fontSize": "200"})
generate_captcha("bar", "out2.png")
