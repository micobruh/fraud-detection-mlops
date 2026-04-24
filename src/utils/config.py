START_DATE = "2017-11-30"
RANDOM_STATE = 42

TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"
DEFAULT_FEATURE_SET = "base_selected_v"
DEFAULT_SEARCH_SMOTE = False
DEFAULT_SEARCH_N_JOBS = 1

BASE_COLUMNS = [
    "TransactionAmt", "ProductCD",
    *[f"card{i}" for i in range(1, 7)],
    *[f"addr{i}" for i in range(1, 3)],
    *[f"dist{i}" for i in range(1, 3)],
    "P_emaildomain", "R_emaildomain",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
    *[f"M{i}" for i in range(1, 10)],
]
V_COLUMNS = [
    "V307", "V284", "V285", "V286", "V299", "V298", "V304", "V305", "V308", "V309", "V310", "V320",
    "V101", "V96", "V98", "V99", "V106", "V105", "V107", "V108", "V109", "V111", "V117", "V120",
    "V121", "V123", "V128", "V127", "V131", "V130", "V134", "V281", "V283", "V289", "V296", "V301",
    "V314", "V12", "V14", "V18", "V20", "V23", "V26", "V27", "V29", "V53", "V56", "V74",
    "V62", "V65", "V67", "V68", "V69", "V75", "V78", "V79", "V83", "V87", "V88", "V89", "V90",
    "V35", "V38", "V52", "V41", "V45", "V47", "V48", "V1", "V3", "V5", "V7", "V9", "V10",
    "V220", "V222", "V234", "V239", "V251", "V271", "V169", "V201", "V175", "V180", "V185",
    "V197", "V210", "V209", "V203", "V173", "V181", "V215", "V205", "V211",
    "V257", "V264", "V223", "V265", "V237", "V240", "V260", "V268", "V275",
    "V324", "V323", "V325", "V330", "V332", "V335",
    "V138", "V140", "V142", "V160", "V147", "V158", "V162"
]
CATEGORICAL_COLUMNS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M1",
    "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "id_12", "id_15", "id_16",
    "id_23", "id_27", "id_28", "id_29", "id_30", "id_31", "id_33", "id_34",
    "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"
]
NUMERICAL_COLUMNS = [
    *[f"card{i}" for i in range(1, 7) if i not in [4, 6]],
    *[f"addr{i}" for i in range(1, 3)],
    *[f"dist{i}" for i in range(1, 3)],
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
    *[f"id_{i}" for i in range(1, 27) if i not in [12, 15, 16, 23]],
    "id_32",
    *V_COLUMNS,
]
DROP_COLUMNS = [
    "TransactionDT", "TransactionID", "D6", "D7", "D8", "D9", "D12", "D13", "D14",
    "C3", "M5", "id_08", "id_33",
    "card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34",
    *["id_" + str(i) for i in range(22, 28)]
]

FEATURE_SETS = {
    "base": {
        "use_selected_v": False,
        "use_uid_features": False,
    },
    "base_selected_v": {
        "use_selected_v": True,
        "use_uid_features": False,
    },
    "base_selected_v_engineered": {
        "use_selected_v": True,
        "use_uid_features": True,
    },
}

UID_COMBINE_COLUMNS = ["card1", "addr1"]
UID_COMBINED_COLUMN = "card1_addr1"
UID_AGGREGATION_MAIN_COLUMNS = ["TransactionAmt"]
UID_AGGREGATION_UID_COLUMNS = [UID_COMBINED_COLUMN, "card1", "addr1"]
UID_AGGREGATION_FUNCTIONS = ["count", "mean", "std", "min", "max"]
