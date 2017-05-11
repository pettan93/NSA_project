def unzip_datasets():
    import os
    for dataset_name in os.listdir("resources/output/"):
        print("Ověřuji %s" % dataset_name)
        zip_path = "resources/output/%s/%s.zip" % (dataset_name, dataset_name)
        if not os.path.exists(zip_path):
            print("Dataset nelze rozbalit, protože %s.zip nebyl nalezen" % dataset_name)
            continue

        import hashlib
        if os.path.exists("resources/output/%s/%s.zip.hash" % (dataset_name, dataset_name)):
            with open(zip_path, "rb") as zip_file:
                hasher = hashlib.sha256()
                data = zip_file.read()
                hasher.update(data)

                with open(zip_path + ".hash", "r") as hash_file:
                    previous_hash = hash_file.read()
                    if previous_hash == hasher.hexdigest():
                        print("%s je v pořádku přeskakuji" % zip_path)
                        continue

        import zipfile
        print("Extrahuji %s" % zip_path)
        try:
            archive = zipfile.ZipFile(zip_path)
            archive.extractall("resources/output/%s/" % dataset_name)
            with open(zip_path, "rb") as zip_file:
                hasher = hashlib.sha256()
                data = zip_file.read()
                hasher.update(data)
                with open(zip_path + ".hash", "w+") as hash_file:
                    hash_file.write(hasher.hexdigest())
        except zipfile.BadZipfile:
            print("Zip je moc velký")
