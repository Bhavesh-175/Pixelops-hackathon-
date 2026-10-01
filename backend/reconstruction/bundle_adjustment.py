def bundle_adjustment_info():

    return {
        "engine": "COLMAP",
        "status": "Bundle adjustment is performed by COLMAP mapper."
    }


if __name__ == "__main__":

    print(bundle_adjustment_info())