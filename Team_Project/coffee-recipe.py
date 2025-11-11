import pandas as pd

# 1️⃣ Load your Excel file
df = pd.read_excel("Data/sales_with_weather_tx.xlsx")

# Normalize column names
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

# Check structure
print(df.head())


# 2️⃣ Define precise recipes based on product type and detail
def detailed_recipe(row):
    ptype = row["product_type"].lower()
    detail = row["product_detail"].lower()

    # ☕ Coffee-based recipes
    if "coffee" in ptype or "espresso" in ptype or "latte" in ptype:
        # Extract bean origin
        if any(x in detail for x in ["ethiopia", "brazilian", "columbian", "jamaican", "guatemalan", "sustainably", "primo", "civet"]):
            bean_origin = [x for x in ["ethiopia", "brazilian", "columbian", "jamaican", "guatemalan", "sustainably", "primo", "civet"] if x in detail][0].capitalize()
        else:
            bean_origin = "House blend"

        # Espresso / latte / drip variations
        if "espresso" in ptype or "shot" in detail:
            return {f"{bean_origin} espresso beans": 8, "Water": 40}
        elif "latte" in ptype or "latte" in detail:
            return {f"{bean_origin} espresso beans": 8, "Milk": 150}
        elif "americano" in ptype:
            return {f"{bean_origin} espresso beans": 8, "Water": 150}
        elif "drip" in ptype or "blend" in detail:
            return {f"{bean_origin} drip coffee grounds": 10, "Water": 180}
        else:
            return {f"{bean_origin} coffee grounds": 8, "Water": 100}

    # 🍵 Chai or tea recipes
    elif "chai" in ptype or "tea" in ptype:
        if "spicy" in detail:
            chai_name = "Spicy Eye Opener Chai"
        elif "morning sunrise" in detail:
            chai_name = "Morning Sunrise Chai"
        elif "traditional" in detail:
            chai_name = "Traditional Blend Chai"
        else:
            chai_name = "Generic Chai"

        if "green" in detail:
            tea_type = "Green tea leaves"
        elif "earl grey" in detail:
            tea_type = "Earl Grey leaves"
        elif "lemon grass" in detail:
            tea_type = "Lemongrass tea leaves"
        elif "english breakfast" in detail:
            tea_type = "English Breakfast leaves"
        else:
            tea_type = "Herbal tea mix"

        if "chai" in ptype:
            return {f"{chai_name} mix": 5, "Milk": 150, "Water": 50, "Sugar": 10}
        else:
            return {tea_type: 3, "Water": 200}

    # 🍫 Chocolate drinks
    elif "chocolate" in ptype or "chocolate" in detail:
        if "dark" in detail:
            return {"Dark chocolate powder": 15, "Milk": 120, "Sugar": 8}
        elif "white" in detail:
            return {"White chocolate powder": 15, "Milk": 120, "Sugar": 8}
        else:
            return {"Cocoa mix": 15, "Milk": 120, "Sugar": 10}

    # 🍰 Pastries
    elif any(x in ptype for x in ["scone", "croissant", "biscotti", "pastry"]):
        return {"Flour": 80, "Butter": 20, "Sugar": 10}

    # 🍯 Syrups
    elif "syrup" in detail:
        if "hazelnut" in detail:
            return {"Hazelnut syrup": 10}
        elif "peppermint" in detail:
            return {"Peppermint syrup": 10}
        elif "caramel" in detail:
            return {"Caramel syrup": 10}
        elif "sugar free vanilla" in detail:
            return {"Sugar-free vanilla syrup": 10}
        else:
            return {"Generic syrup": 10}

    # 🎁 Non-edibles
    elif any(x in detail for x in ["t-shirt", "mug", "houseware"]):
        return {"Non-consumable": 0}

    # Default case
    else:
        return {"Unknown": 0}


# 3️⃣ Size multiplier
def size_multiplier(detail):
    detail = detail.lower()
    if "sm" in detail:
        return 1.0
    elif "rg" in detail or "md" in detail or "medium" in detail:
        return 1.25
    elif "lg" in detail or "large" in detail:
        return 1.5
    elif "xl" in detail or "extra" in detail:
        return 1.75
    else:
        return 1.0


# 4️⃣ Expand rows into detailed ingredient quantities
expanded = []

for _, row in df.iterrows():
    recipe = detailed_recipe(row)
    scale = size_multiplier(row["product_detail"])
    for ingredient, qty in recipe.items():
        expanded.append({
            "product_type": row["product_type"],
            "product_detail": row["product_detail"],
            "ingredient": ingredient,
            "quantity_per_recipe": qty * scale
        })

expanded_df = pd.DataFrame(expanded)

# 5️⃣ Summarize total quantities by ingredient
inventory = expanded_df.groupby("ingredient", as_index=False)["quantity_per_recipe"].sum()

print("\n--- Detailed Inventory by Ingredient ---")
print(inventory)

# 6️⃣ Export both detailed and summarized results
expanded_df.to_excel("/Users/morganenaibo/Downloads/detailed_inventory_precise.xlsx", index=False)
inventory.to_excel("/Users/morganenaibo/Downloads/total_inventory_precise.xlsx", index=False)
