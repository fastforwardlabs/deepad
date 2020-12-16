/*
 * ****************************************************************************
 *
 *  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
 *  (C) Cloudera, Inc. 2020
 *  All rights reserved.
 *
 *  Applicable Open Source License: Apache 2.0
 *
 *  NOTE: Cloudera open source products are modular software products
 *  made up of hundreds of individual components, each of which was
 *  individually copyrighted.  Each Cloudera open source product is a
 *  collective work under U.S. Copyright Law. Your license to use the
 *  collective work is as provided in your written agreement with
 *  Cloudera.  Used apart from the collective work, this file is
 *  licensed for your use pursuant to the open source license
 *  identified above.
 *
 *  This code is provided to you pursuant a written agreement with
 *  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
 *  this code. If you do not have a written agreement with Cloudera nor
 *  with an authorized and properly licensed third party, you do not
 *  have any rights to access nor to use this code.
 *
 *  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
 *  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
 *  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
 *  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
 *  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
 *  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
 *  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
 *  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
 *  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
 *  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
 *  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
 *  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
 *  DATA.
 *
 * ***************************************************************************
 */

// Web interface to visualize example Anomaly Detection Interaction.
// Data is visualized using the Carbon Desgin System Data Table.
// Data is

import React, { Component } from "react";
import { DataTable, InlineLoading } from "carbon-components-react";
import {
  getJSONData,
  probabilityColor,
  abbreviateString,
  postJSONData,
} from "../helperfunctions/HelperFunctions";
import "./groupview.css";
import DetailView from "../detailview/DetailView";
const {
  Table,
  TableHead,
  TableHeader,
  TableBody,
  TableCell,
  TableRow,
} = DataTable;

class GroupView extends Component {
  constructor(props) {
    super(props);

    this.state = {
      visibleColumns: 12,
      numDataRows: 300,
      visibleRows: 300,
      // stickyHeader: true,
      tableTitle: " ",
      tableIsSortable: false,
      tableSize: "normal", //tall short normal
      dataRows: [],
      columnNames: ["id"],
      columnDescription: {},
      targetFeature: "",
      datasetName: "KDD 99, Intrusion Detection",
      cellColors: {},
      selecetedRowid: 0,
      predictionsLoaded: false,
      dataLoaded: false,
      showTableView: true,
      maxNumericLength: 7,
      tableHeight: "600px",
    };

    this.baseUrl = window.location.protocol + "//" + window.location.host;
    this.dataEndpoint = "/data";
    this.predictEndpoint = "/predict";

    this.colnameEndpoint = "/colnames";
    this.hideDetailView = this.hideDetailView.bind(this);

    this.predictionTitle = "Model Prediction";
  }

  // Hide detail view
  hideDetailView() {
    this.setState({ showTableView: true, showDetailView: false });
  }

  documentHeight() {
    return Math.max(
      document.documentElement.clientHeight,
      document.body.scrollHeight,
      document.documentElement.scrollHeight,
      document.body.offsetHeight,
      document.documentElement.offsetHeight
    );
  }

  componentDidMount() {
    // Fetch Feature List / Data Header
    let getDataHeaderURL = this.baseUrl + this.colnameEndpoint;
    let colnames = getJSONData(getDataHeaderURL);
    colnames.then((data) => {
      if (data) {
        let colnames = data["colnames"];
        let coldesc = data["coldesc"];
        if (!colnames.includes("id")) {
          colnames.unshift("id");
          coldesc.unshift("id");
        }
        // Add target label to headers
        colnames.unshift(data["label"]);
        coldesc.unshift(data["label"]);

        // Add prediction label to headers
        colnames.unshift(this.predictionTitle);
        coldesc.unshift(this.predictionTitle);

        this.setState({
          columnNames: colnames,
          targetFeature: data["label"],
          columnDescription: coldesc,
        });
        this.loadData();
      }
    });

    let tableHeight =
      this.documentHeight() -
      10 -
      document.getElementsByClassName("tablecontent")[0].getBoundingClientRect()
        .top;

    this.setState({ tableHeight: tableHeight + "px" });
  }

  // Load Data from Model End Point
  loadData() {
    this.setState({ cellColors: {} });
    // Fetch Data after features have arrived
    let getDataURL =
      this.baseUrl + this.dataEndpoint + "?n=" + this.state.numDataRows;
    let data = getJSONData(getDataURL);
    data.then((data) => {
      //Create Colors for Target Column
      let cellColors = {};
      // Datable requires string id
      for (let [i, row] of data.entries()) {
        if (row["id"]) {
          row["id"] = row["id"] + "";
        } else {
          row["id"] = i + "";
        }
        cellColors[
          row["id"] + ":" + this.state.targetFeature
        ] = probabilityColor(row[this.state.targetFeature]);
      }
      this.setState({
        dataRows: data,
        numDataRows: data.length,
        cellColors: cellColors,
        dataLoaded: true,
      });
      this.getPredictions(data);
    });
  }

  // Make requests to model endpoint to get predictions
  getPredictions(data) {
    data = data.slice(0, this.state.visibleRows);
    let predictURL = this.baseUrl + this.predictEndpoint;
    let predictions = postJSONData(predictURL, { data: data });
    let cellColors = this.state.cellColors;
    predictions.then((data) => {
      let currentData = this.state.dataRows;

      for (let [i, prediction] of data["predictions"].entries()) {
        currentData[i][this.predictionTitle] = prediction;
        cellColors[
          data["ids"][i] + ":" + this.predictionTitle
        ] = probabilityColor(prediction);
      }
      this.setState({
        dataRow: currentData,
        cellColors: cellColors,
        predictionsLoaded: true,
      });
    });
  }

  // handle row click event
  clickRow(e) {
    this.setState({
      selecetedRowid: e.target.getAttribute("rowindex"),
      showTableView: false,
      showDetailView: true,
    });
  }

  render() {
    let headers = this.state.columnNames
      .slice(0, this.state.visibleColumns)
      .map((data, index) => {
        return {
          key: data,
          header: this.state.columnDescription[index] || data,
        };
      });

    // Add elispsis if we arent showing all feature columns
    if (this.state.columnNames.length >= this.state.visibleColumns) {
      headers.push({ key: "...", header: "..." });
    }

    let rows = this.state.dataRows
      .slice(0, this.state.visibleRows)
      .map((data, index) => {
        let dataRow = {};
        for (let feature of this.state.columnNames.slice(
          0,
          this.state.numShow
        )) {
          let featureValue = data[feature] === undefined ? "_" : data[feature];
          dataRow[feature] = abbreviateString(
            featureValue + "",
            this.state.maxNumericLength
          );
        }
        return dataRow;
      });

    let currentDataDetails = [];

    if (this.state.dataLoaded) {
      let row = this.state.dataRows[this.state.selecetedRowid];
      currentDataDetails = [];
      for (let [i, key] of this.state.columnNames.entries()) {
        currentDataDetails.push({
          id: row["id"],
          feature: this.state.columnDescription[i],
          value: row[key],
        });
      }
    }

    return (
      <div>
        <div className="boldtext sectiontitle p10">
          Anomaly Detection on Network Intrusion Data
        </div>
        <div className="mynotif mt10 h100 lh10  lightgreyhighlight p10 maxh16  mb10">
          The{" "}
          <a
            href="http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html"
            target="black"
          >
            KDD network intrusion
          </a>{" "}
          dataset is a dataset of TCP connections that have been labeled as
          normal or representative of network attacks. Each TCP connection is
          represented as a set of attributes or features (derived based on
          domain knowledge) pertaining to each connection such as the number of
          failed logins, connection duration, data bytes from source to
          destination, etc. The table below is a random sample of{" "}
          <strong> {this.state.visibleRows}</strong> test data points which are
          being classified as normal or abnormal by a trained autoencoder model.
          The original ground truth label (label) as well as the prediction by
          the model is shown.
        </div>
        <div className="flex">
          {!this.state.predictionsLoaded && (
            //
            <div className="smalldesc   iblock flex">
              <div className="iblock   mr5">
                {" "}
                <InlineLoading></InlineLoading>{" "}
              </div>
              <div className="iblock   flex flexcolumn flexjustifycenter">
                {" "}
                loading anomaly predictions ...{" "}
              </div>
            </div>
          )}
          {this.state.predictionsLoaded && (
            <div className="smalldesc p10  flex flexcolumn flexjustifycenter">
              Showing{" "}
              {Math.min(
                this.state.visibleColumns,
                this.state.columnNames.length
              )}{" "}
              of {this.state.columnNames.length} columns{" "}
              {this.state.visibleRows} rows.{" "}
            </div>
          )}
        </div>

        <div className="positionrelative  ">
          <div className="positionabsolute   w100">
            {this.state.dataLoaded && (
              <div className={this.state.showDetailView ? "" : "displaynone"}>
                <DetailView
                  dataDetails={currentDataDetails}
                  targetFeature={this.state.targetFeature}
                  cellColors={this.state.cellColors}
                  hideDetail={this.hideDetailView}
                  targetFeatureValue={
                    this.state.dataRows[this.state.selecetedRowid][
                      this.state.targetFeature
                    ]
                  }
                ></DetailView>
              </div>
            )}
          </div>

          {
            <div
              className={
                this.state.showTableView
                  ? "mb10  datatable-body"
                  : "displaynone"
              }
            >
              <DataTable
                isSortable={this.state.tableIsSortable}
                rows={rows}
                headers={headers}
                render={({ rows, headers, getHeaderProps }) => (
                  // <TableContainer title={this.state.tableTitle + this.state.datasetName}>
                  <Table
                    style={{ minHeight: this.state.tableHeight }}
                    className=" tablecontent "
                    stickyHeader={this.state.stickyHeader}
                    size={this.state.tableSize}
                  >
                    <TableHead>
                      <TableRow>
                        {headers.map((header) => (
                          <TableHeader {...getHeaderProps({ header })}>
                            {header.header}
                          </TableHeader>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {rows.map((row, index) => (
                        <TableRow
                          style={{ cursor: "pointer" }}
                          onClick={this.clickRow.bind(this)}
                          key={row.id}
                          rowid={row.id}
                        >
                          {row.cells.map((cell) => (
                            <TableCell
                              style={{
                                backgroundColor: this.state.cellColors[cell.id],
                              }}
                              key={cell.id}
                              rowid={row.id}
                              rowindex={index}
                            >
                              {cell.value}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  // </TableContainer>
                )}
              />
            </div>
          }
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
