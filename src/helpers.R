library(rfishbase)
library(dplyr)
library(readr)

load_species_list <- function(input_file) {
    read_csv(input_file)
}

get_diet_data <- function(species_list) {
    diet(species_list$`B.species`)
}

get_diet_items_data <- function() {
    diet_items()
}

merge_diet_data <- function(diet_data, diet_items_data) {
    diet_data %>% left_join(diet_items_data, by = "DietCode")
}

process_and_save_data <- function(species_list, merged_data, output_file) {
    required_columns <- c("Species", "Troph", "FoodI", "FoodII", "FoodIII", "Stage", "DietPercent", "ItemName")

    if (all(required_columns %in% colnames(merged_data))) {
        final_data <- species_list %>%
            left_join(merged_data, by = c("B.species" = "Species")) %>%
            select(`B.class`, `B.family`, `B.species`, Troph, FoodI, FoodII, FoodIII, Stage, DietPercent, ItemName, `ID`) %>%
            rename(Class = `B.class`, Family = `B.family`, Species = `B.species`)

        write_csv(final_data, output_file)
        return(TRUE)
    } else {
        return(FALSE)
    }
}

main <- function(input_file, output_file) {
    species_list <- load_species_list(input_file)
    diet_data <- get_diet_data(species_list)
    diet_items_data <- get_diet_items_data()
    merged_data <- merge_diet_data(diet_data, diet_items_data)

    success <- process_and_save_data(species_list, merged_data, output_file)

    if (success) {
        print(paste("Output saved to", output_file))
    } else {
        print("One or more required columns are missing in the merged_data dataframe. Please check the column names.")
    }
}
