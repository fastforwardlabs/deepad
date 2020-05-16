import React, { Component } from "react";
import { DataTable, InlineLoading } from 'carbon-components-react';
import * as _ from "lodash"
import { getJSONData, ColorExplanation, probabilityColor } from "../helperfunctions/HelperFunctions"
import "./groupview.css"
import DetailView from "../detailview/DetailView";

const { Table, TableHead, TableHeader, TableBody, TableCell, TableRow } = DataTable;


class GroupView extends Component {
    constructor(props) {
        super(props)

        this.state = {
            visibleColumns: 9,
            visibleRows: 20,
            stickyHeader: false,
            tableTitle: " ",
            tableIsSortable: false,
            tableSize: "normal", //tall short normal
            dataRows: [],
            columnNames: ["id"],
            columnDescription: {},
            explainMethod: "LIME",
            targetFeature: "",
            datasetName: "IBM Churn Dataset",
            numDataRows: 100,
            cellColors: {},
            selecetedRowid: 0,
            loadedExplanationCount: 0,
            explanationLoaded: false,
            dataLoaded: false,
            showDetailView: false,
            showTableView: true,
        }

        this.baseUrl = "http://localhost:5000"
        this.dataEndpoint = "/data"
        this.colnameEndpoint = "/colnames"

        this.explanations = {}


        this.hideDetailView = this.hideDetailView.bind(this);

    }
    componentDidMount() {
        // Fetch Feature List / Data Header
        let getDataHeaderURL = this.baseUrl + this.colnameEndpoint
        let colnames = getJSONData(getDataHeaderURL)
        colnames.then((data) => {
            if (data) {
                let colnames = data["colnames"]
                let coldesc = data["coldesc"]
                if (!colnames.includes("id")) {
                    colnames.unshift("id")
                    coldesc.unshift("id")
                }
                // Add target label to headers
                colnames.unshift(data["label"])
                coldesc.unshift(data["label"])

                this.setState({ columnNames: colnames, targetFeature: data["label"], columnDescription: coldesc })
                this.loadData()


            }
        })
    }

    loadData() {
        // Clear Cell Colors Dict
        this.setState({ cellColors: {} })
        // Fetch Data after features have arrived
        let getDataURL = this.baseUrl + this.dataEndpoint + "?N=" + this.state.numDataRows
        let data = getJSONData(getDataURL)
        data.then((data) => {
            //Create Colors for Target Column 
            let cellColors = {}
            // Datable requires string id
            for (let [i, row] of data.entries()) {
                if (row["id"]) {
                    row["id"] = row["id"] + "";
                } else {
                    row["id"] = i + "";
                }
                cellColors[row["id"] + ":" + this.state.targetFeature] = probabilityColor(row[this.state.targetFeature])
            }
            this.setState({
                dataRows: data, numDataRows: data.length, cellColors: cellColors,
                dataLoaded: true
            })
            // this.getExplanations(data)


        })
    }

    getExplanations(data) {
        for (let row of data.slice(0, this.state.visibleRows)) {
            let queryString = "?"
            for (var key in row) {
                if (row.hasOwnProperty(key)) {
                    queryString += "&" + key + "=" + encodeURIComponent(row[key])
                }
            }
            this.getExplanation(row["id"], queryString)
        }
    }




    getExplanation(dataId, queryString) {
        let explainURL = this.baseUrl + "/explain" + queryString
        let explanation = getJSONData(explainURL)

        explanation.then((data) => {
            if (data) {
                this.explanations[dataId] = data["explanation"]
                this.updateCellColor(dataId, data["explanation"])
                // console.log(data["explanation"], this.state.loadedExplanationCount);
                this.setState({ loadedExplanationCount: this.state.loadedExplanationCount + 1 }, () => {
                    if (this.state.loadedExplanationCount === this.state.visibleRows) {
                        this.setState({ explanationLoaded: true })
                    }
                })


            } else {
                console.log("Failed to fetch explanation");

            }


        })
    }

    updateCellColor(dataId, row) {
        // Set color of each feature based on a gradation on from its min to max
        let cellColors = this.state.cellColors
        let rowMin = _.min(Object.values(row))
        let rowMax = _.max(Object.values(row))

        for (var key in row) {
            if (row.hasOwnProperty(key)) {
                cellColors[dataId + ":" + key] = ColorExplanation(rowMin, rowMax, row[key])

            }

        }

        this.setState({ cellColors: cellColors })

    }

    clickRow(e, f) {
        this.setState({ selecetedRowid: e.target.getAttribute("rowindex"), showTableView: false, showDetailView: true })
    }

    hideDetailView() {
        this.setState({ showTableView: true, showDetailView: false })
    }


    render() {
        let headers = this.state.columnNames.slice(0, this.state.visibleColumns).map((data, index) => {
            return ({ key: data, header: this.state.columnDescription[index] || data })
        });

        // Add elispsis if we arent showing all feature columns
        if (this.state.columnNames.length > this.state.visibleColumns) {
            headers.push({ key: "...", header: "..." })
        }

        let rows = this.state.dataRows.slice(0, this.state.visibleRows).map((data, index) => {
            let dataRow = {}
            for (let feature of this.state.columnNames.slice(0, this.state.numShow)) {
                dataRow[feature] = data[feature]
            }
            return (dataRow)
        });

        let fullExplanation = []

        if (this.state.dataLoaded && this.state.explanationLoaded) {
            let row = this.state.dataRows[this.state.selecetedRowid];
            fullExplanation = []
            for (let key of Object.keys(row)) {
                fullExplanation.push({ id: row["id"], feature: key, value: row[key], explanation: this.explanations[row["id"]][key] || 0 })
            }
            // sort and show by explanation value
            fullExplanation = _.sortBy(fullExplanation, o => {
                return o.explanation === 0 ? Infinity : o.explanation
            })
        }


        return (
            <div>

                <div className="boldtext sectiontitle p10">
                    Anomaly Detection on Network Intrusion Data
                </div>
                <div className="flex">

                    {!this.state.explanationLoaded &&
                        //
                        <div className="smalldesc   iblock flex">
                            <div className="iblock   mr5"> <InlineLoading></InlineLoading>  </div>
                            <div className="iblock   flex flexcolumn flexjustifycenter"> loading explanations .. {this.state.loadedExplanationCount} of {this.state.visibleRows} </div>

                        </div>}
                    {this.state.explanationLoaded && <div className="smalldesc p10  flex flexcolumn flexjustifycenter">Showing  {Math.min(this.state.visibleColumns, this.state.columnNames.length)} features of {this.state.visibleRows}  row. </div>}

                </div>
                <div style={{ width: Math.round(this.state.loadedExplanationCount / this.state.visibleRows * 100) + "%" }} className="glowbar mb5"></div>


                <div className="positionrelative">
                    <div className="positionabsolute  w100">
                        {(this.state.dataLoaded && this.state.explanationLoaded && this.state.showDetailView) &&
                            <div className=" ">
                                <DetailView
                                    explanation={fullExplanation}
                                    targetFeature={this.state.targetFeature}
                                    cellColors={this.state.cellColors}
                                    hideDetail={this.hideDetailView}
                                    targetFeature={this.state.targetFeature}
                                    targetFeatureValue={this.state.dataRows[this.state.selecetedRowid][this.state.targetFeature]}
                                ></DetailView>
                            </div>
                        }
                    </div>
                    {this.state.showTableView && <div className=" mb10  datatable-body">
                        <DataTable
                            isSortable={this.state.tableIsSortable}
                            rows={rows}
                            headers={headers}
                            render={({ rows, headers, getHeaderProps }) => (
                                // <TableContainer title={this.state.tableTitle + this.state.datasetName}>
                                <Table stickyHeader={this.state.stickyHeader} size={this.state.tableSize}>
                                    <TableHead>
                                        <TableRow>
                                            {headers.map(header => (
                                                <TableHeader {...getHeaderProps({ header })}>
                                                    {header.header}
                                                </TableHeader>
                                            ))}
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {rows.map((row, index) => (
                                            <TableRow style={{ cursor: "pointer" }} onClick={this.clickRow.bind(this)} key={row.id} rowid={row.id}>
                                                {row.cells.map(cell => (
                                                    <TableCell style={{ backgroundColor: this.state.cellColors[cell.id] }} key={cell.id} rowid={row.id} rowindex={index}>{cell.value}</TableCell>
                                                ))}
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                                // </TableContainer>
                            )}
                        />
                    </div>}

                </div>





                <div>
                    {/* {JSON.stringify(this.state.dataRows)} */}
                    {/* {explanationColor(0.1) + "\t" + JSON.stringify(this.state.cellColors)} */}
                </div>


            </div>

        );
    }
}

export default GroupView;