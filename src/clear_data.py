import pandas as pd

def get_columns(path_list):

  columns = ["Equipe"]

  for path in path_list:

    with open(path, "r", encoding = "utf-8-sig") as file:
      
      for line in file:

        if line.strip() == "":

          pass

        elif ":" in line:

          line_content = line.split(":")
          if line_content[0] not in columns:
            columns.append(line_content[0])

  return columns

def create_dataset(path_list, columns = None):

  columns = columns if columns != None else get_columns(path_list)

  df = pd.DataFrame(columns = columns)

  row_df = 0

  for path in path_list:

    Equipe = path.split("_")[0]
    print(Equipe)

    with open(path, "r", encoding = "utf-8-sig") as file:

      is_data = True

      for i, line in enumerate(file):

        if line.strip() == "":

          is_data = False


        elif ":" in line:

          if is_data == False:

            row_df += 1
            is_data = True

          line_content = line.split(":")
          column = line_content[0]
          content = ":".join(line_content[1:]).strip()
          
        else:

          content += "," + line 
            

        df.loc[row_df, column] = content
        df.loc[row_df, "Equipe"] = Equipe

    file.close()

  return df

if __name__ == "__main__":
  
    path_list = ["Team_17_12_2024.txt", "SAFT_17_12_2024.txt", "Interet_17_12_2024.txt"]
    columns = get_columns(path_list)


    #ordre plus sympas
    columns = ['Equipe', 'Reference Type', 'Record Number', 'Author', 'Year', 'Title', 'Short Title',
    'Journal', 'Volume', 'Issue', 'Date', 'ISSN', 'DOI', 'Article Number', 'Accession Number',
    'NIHMSID', 'Conference Name', 'Pages', 'Editor', 'Book Title', 'ISBN', 'Series Title', 'Number of Pages', 'Abstract']

    df = create_dataset(path_list, columns)

    #sep = ";" car des virgules un peu partout dans les abstracts
    df.to_csv('abstracts.csv', index=False, sep = ";")