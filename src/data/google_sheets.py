# google_sheets.py
# Matthew J Cronin, 2020
# matthew.j.cronin@gmail.com
#
# Useful tools for interfacing with Google Sheets from python.

from __future__ import print_function
import pickle
import os.path
import sys

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def get_creds(read_write='read', creds_path='~/.credentials/google_sheets/'):
    """
    Returns credentials for Google Sheets API
    
    Modified from code found at https://developers.google.com/sheets/api/quickstart/python

    Keyword Arguments:
        read_write (str): "read" if reading data or "write" if writing data
    
    Returns:
        creds: credentials for Google Sheets API
    """
    creds_path = os.path.expanduser(creds_path)
    
    if read_write == 'read':
        # If modifying these scopes, delete the file token.pickle.
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.read_only']
        creds_dir = 'sheets_read'
    elif read_write == 'write':
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.read_only']
        creds_dir = 'sheets_write'
    else:
        raise Exception('read_write must have a value of "read" or "write"')

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(creds_path+'/{}/token.pickle'.format(creds_dir)):
        with open(creds_path+'/{}/token.pickle'.format(creds_dir), 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            #try:
            print('path = {}'.format(creds_path+'credentials.json'))
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path+'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            #except:>.
            #   print('Could not load {}credentials.json - check path and try again.'.format(creds_path))
             #  sys.exit()
            
            # Save the credentials for the next run
            if not os.path.exists(creds_path+'{}/'.format(creds_dir)):
                os.mkdir(creds_path+'{}/'.format(creds_dir))
            with open(creds_path+'/{}/token.pickle'.format(creds_dir), 'wb') as token:
                pickle.dump(creds, token)
    
    return creds


def gs_read(spreadsheet_ID, cell_range,creds_path='~/.credentials/google_sheets/'):
    """
    Reads and returns data from Google sheet in specified range of cells.
       
       Modified from code found at https://developers.google.com/sheets/api/quickstart/python
    
    Keyword Arguments:
        spreadsheet_ID (str): Unique spreadsheet_id taken from Google sheet URL
               docs.google.com/spreadsheets/d/spreadsheet_id/...
        cell_range (str): Range of cells to be read from sheet e.g 'A1:C5'
        creds_path (str): Path to credentials folder.  - if unavailable this can be obtained at the URL above.

    Returns:
        array: Contents of cells.
    """
    creds = get_creds(read_write='read', creds_path=creds_path)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_ID,
                                range=cell_range).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        return(values)


def gs_write(data, spreadsheet_ID, cell_range, majorDimension='ROWS', valueInputOption='RAW', creds_path='~/.credentials/google_sheets/'):
    """
    Writes data to Google sheet in specified range of cells.
       
       Modified from code found at https://developers.google.com/sheets/api/quickstart/python
    
    Keyword Arguments:
        data (array): Data to be written.
        spreadsheet_id (str): Unique spreadsheet_id taken from Google sheet URL
               docs.google.com/spreadsheets/d/spreadsheet_id/...
        cell_range (str): Range of cells to be written to in A1 notation e.g 'Sheet1!A1:C5'
        ValueInputOption (str): "RAW" - store input data as-is
                                  "USER_ENTERED" - parse input data as if typed
                                  directly into spreadsheet
        creds_path (str): Path to credentials folder.  - if unavailable this can be obtained at the URL above.

    Returns:
        None
    """
    creds = get_creds('write', creds_path=creds_path)
    
    service = build('sheets', 'v4', credentials=creds)

    body = {
            'values': data,
            'majorDimension': majorDimension
            }

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().update(
            spreadsheetId=spreadsheet_ID, range=cell_range,
            valueInputOption=valueInputOption, body=body
            ).execute()

    return result


def add_sheet(spreadsheetId, new_sheet, creds_path='~/.credentials/google_sheets/'):
    """
    Writes data to Google sheet starting at specified cell.
       
       Modified from code found at https://developers.google.com/sheets/api/quickstart/python
    
    Keyword Arguments:
        sheet_ID (str): Unique sheet_id taken from Google sheet URL
               docs.google.com/spreadsheets/d/sheet_id/...
        new_sheet (str): Name of new sheet to add to spreadsheet
        creds_path (str): Path to credentials folder.  - if unavailable this can be obtained at the URL above.

    Returns:
        None

    """
    creds = get_creds(read_write='write', creds_path=creds_path)

    service = build('sheets', 'v4', credentials=creds)
    
    # Package changes to be applied to sheet
    requests = []
    addSheet = {
            'addSheet': {
                'properties': {
                    'title': new_sheet
                    }
                }
            }
    requests.append(addSheet)
    
    body = {
            'requests': requests
            }

    # Call the Sheets API
    sheet = service.spreadsheets()
    response = sheet.batchUpdate(spreadsheetId=spreadsheetId,
                                body=body).execute()
    print(response)


def main():
    sheet_id=sys.argv[1]
    cell_range=sys.argv[2]

    read(sheet_id, cell_range,creds_path='~/.credentials/google_sheets/')

if __name__ =='__main__':
    main()






